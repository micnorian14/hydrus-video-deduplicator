"""--rehash-overhashed (off by default): find videos hashed with far too many frames and hash them again."""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import print
from rich.text import Text

from .dedup import TQDM_SMOOTHING, FileHasher, PerceptuallyHashedFile, ProgressBar
from .dedup_util import STYLE_WARN_LINE, console_print, print_and_log, print_success, print_user

if TYPE_CHECKING:
    from collections.abc import Callable

    from .client import HVDClient
    from .db import DedupeDB

log = logging.getLogger("rehash")

# Normal hashing keeps about 1 frame per second of video. A video is over-hashed if it has at least this many hashed
# frames AND more than this many per second of its length (Hydrus' duration). Correctly hashed videos stay near 1.
OVERHASH_MIN_FRAMES = 30
OVERHASH_FRAMES_PER_SECOND = 3.0
METADATA_BATCH_SIZE = 256


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


def is_overhashed(frames: int, duration_ms: float | None) -> bool:
    """True if a video with this many hashed frames and this length (milliseconds) was hashed with too many frames."""
    if not duration_ms or duration_ms <= 0 or frames < OVERHASH_MIN_FRAMES:
        return False
    return frames / (duration_ms / 1000) > OVERHASH_FRAMES_PER_SECOND


def hydrus_durations(client: HVDClient, file_hashes: list[str]) -> dict[str, float | None]:
    """Video lengths in milliseconds from Hydrus (read only: GET /get_files/file_metadata). None if unknown."""
    metadata = client.client.get_file_metadata(hashes=file_hashes, only_return_basic_information=True)
    return {str(m.get("hash", "")).lower(): m.get("duration") for m in metadata.get("metadata", [])}


def find_overhashed(
    candidates: list[tuple[str, int]],
    get_durations: Callable[[list[str]], dict[str, float | None]],
    progress: Callable[[int], None] | None = None,
) -> list[tuple[str, int, float]]:
    """(file hash, hashed frames, duration in ms) for each over-hashed candidate (file hash, hashed frames)."""
    found = []
    for start in range(0, len(candidates), METADATA_BATCH_SIZE):
        batch = candidates[start : start + METADATA_BATCH_SIZE]
        durations = get_durations([file_hash for file_hash, _ in batch])
        for file_hash, frames in batch:
            duration = durations.get(file_hash)
            if is_overhashed(frames, duration):
                found.append((file_hash, frames, duration))
        if progress:
            progress(len(batch))
    return found


@dataclass
class RehashResult:
    found: int = 0
    rehashed: int = 0
    unchanged: int = 0  # hashed again with the same result, so nothing was changed
    failed: int = 0  # keep their old hash
    old_frames: int = 0  # of the re-hashed videos
    new_frames: int = 0


def rehash_message(result: RehashResult) -> str:
    if result.rehashed == 0:
        return "No over-hashed videos were changed."
    return (
        f"Re-hashed {result.rehashed} over-hashed {_plural(result.rehashed, 'video')} "
        f"({result.old_frames} hashed frames down to {result.new_frames}). "
        "They are added to the search tree in this run."
    )


def run_rehash_overhashed(db: DedupeDB.DedupeDb, client: HVDClient, job_count: int = -2) -> RehashResult | None:
    """
    --rehash-overhashed: re-hash videos with far more hashed frames than their length explains (take a backup first).

    Only reads from Hydrus: video lengths and the files. Failed files keep their old hash and are not sent to the
    failed page. All-or-nothing: on an error or CTRL+C the database is left unchanged. Returns None if nothing was
    checked.
    """
    print("[bright_blue]Checking for over-hashed videos (hashed with too many frames)...")
    candidates = db.get_hashed_frame_counts(OVERHASH_MIN_FRAMES)
    result = RehashResult()
    try:
        with ProgressBar(
            smoothing=TQDM_SMOOTHING,
            total=len(candidates),
            desc="Checking video lengths",
            dynamic_ncols=True,
            unit="video",
            colour="GREEN",
        ) as pbar:
            found = find_overhashed(candidates, lambda hashes: hydrus_durations(client, hashes), pbar.update)
    except KeyboardInterrupt:
        print("[yellow]Checking for over-hashed videos was interrupted! Nothing was changed.")
        return None
    except Exception as exc:
        reason = re.sub(r"[0-9a-fA-F]{64}", "<hidden>", str(exc))  # never show an API key or file hash
        print_and_log(
            log,
            f"Skipping --rehash-overhashed: could not get video lengths from Hydrus ({reason}).",
            logging.WARNING,
            highlight=False,
        )
        return None
    result.found = len(found)
    if not found:
        print_success("No over-hashed videos found.")
        return result

    new_phashes: dict[str, bytes] = {}
    filehasher = FileHasher(client, job_count)
    try:
        with ProgressBar(
            smoothing=TQDM_SMOOTHING,
            total=len(found),
            desc="Re-hashing over-hashed videos",
            dynamic_ncols=True,
            unit="video",
            colour="GREEN",
        ) as pbar:
            for file_hash, frames, _ in found:
                hashed = filehasher.fetch_and_phash_file(file_hash)
                gc.collect()
                pbar.update(1)
                if not isinstance(hashed, PerceptuallyHashedFile):
                    result.failed += 1
                    with pbar.external_write_mode(file=pbar.fp):
                        print("[red]Failed to re-hash a file. It keeps its old hash.")
                        print(f"[red]Failed file SHA256 hash: {file_hash}")
                        console_print(Text(str(hashed.exc)), highlight=False)
                    continue
                if hashed.perceptual_hash == db.get_stored_phash(file_hash):
                    result.unchanged += 1
                    continue
                new_phashes[file_hash] = hashed.perceptual_hash
                result.old_frames += frames
                result.new_frames += len(hashed.perceptual_hash) // 32
    except KeyboardInterrupt:
        print("[yellow]Re-hashing was interrupted! Nothing was changed.")
        return None

    if new_phashes:
        try:
            db.begin_transaction()
            with db.conn:  # all-or-nothing
                db.replace_perceptual_hashes(new_phashes)
        except KeyboardInterrupt:
            print("[yellow]Re-hashing was interrupted! Nothing was changed.")
            return None
    result.rehashed = len(new_phashes)
    log.info(f"Re-hash over-hashed: {result}")
    print_success(rehash_message(result))
    if result.unchanged:
        print_user(
            f"{result.unchanged} {_plural(result.unchanged, 'video')} hashed the same as before, so "
            f"{_plural(result.unchanged, 'it was', 'they were')} left unchanged.",
            highlight=False,
        )
    if result.failed:
        print_user(
            f"{result.failed} {_plural(result.failed, 'video')} could not be re-hashed and "
            f"{_plural(result.failed, 'keeps its', 'keep their')} old hash.",
            default_style=STYLE_WARN_LINE,
            highlight=False,
        )
    return result
