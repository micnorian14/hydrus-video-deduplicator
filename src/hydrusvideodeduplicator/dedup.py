from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich import print
from rich.text import Text
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypeAlias

    FileHash = str

import atexit
import contextlib
import gc
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import hydrusvideodeduplicator.hydrus_api as hydrus_api

from .client import HVDClient
from .db import DedupeDB, vptree
from .dedup_util import console_print, memory_status, print_success, print_user, style_user_text
from .config import GPU_DECODE_AUTO
from .hashing import compute_phash, last_decode_used_gpu, video_prefers_gpu_decode
from .page_logger import HydrusPageLogger, resolve_failed_page_name
from hvdaccelerators import vpdq


@dataclass
class PerceptuallyHashedFile:
    """Class for perceptually hashed files."""

    file_hash: FileHash
    perceptual_hash: bytes


@dataclass
class FailedPerceptuallyHashedFile:
    """Class for failed perceptually hashed files."""

    file_hash: FileHash
    exc: Exception


# Progress bar speed/ETA smoothing (display only). tqdm's default 0.3 makes the speed jump with every slow or fast
# file; 0.05 averages over roughly the last ~20 updates, so it still follows real changes but swings much less.
TQDM_SMOOTHING = 0.05

# Progress bars never get wider than this (display only). They still shrink with narrower windows and re-measure
# the window on every refresh (dynamic_ncols). 110 fits a console on half of a 1080p screen, and keeps the bar
# short enough that resizing the window doesn't smear old bar lines as much.
MAX_BAR_COLS = 110

# Display-only colours for progress bar stats (ANSI). tqdm's disp_len strips these, so the 110-col cap
# still measures visible width. Cyan matches Rich's number style (bold cyan). Bright yellow for times/rate.
_BAR_CYAN = "\x1b[1;36m"
_BAR_YELLOW = "\x1b[93m"
_BAR_RESET = "\x1b[0m"

# Known total: "  8%|...| 62/219690 [00:06<5:56:35, 10.27video/s]"
BAR_FORMAT = (
    "{desc}: "
    f"{_BAR_CYAN}{{percentage:3.0f}}%{_BAR_RESET}"
    "|{bar}| "
    f"{_BAR_CYAN}{{n_fmt}}{_BAR_RESET}/{_BAR_CYAN}{{total_fmt}}{_BAR_RESET}"
    f" [{_BAR_YELLOW}{{elapsed}}{_BAR_RESET}<{_BAR_YELLOW}{{remaining}}{_BAR_RESET}, "
    f"{_BAR_YELLOW}{{rate_fmt}}{_BAR_RESET}{{postfix}}]"
)

# Unknown or zero total: "Building search tree: 0 videos [00:00, ?video/s]"
# Space + plural "videos" for the count only; the rate unit stays "video/s".
BAR_FORMAT_NO_TOTAL = (
    "{desc}: "
    f"{_BAR_CYAN}{{n_fmt}}{_BAR_RESET} videos "
    f"[{_BAR_YELLOW}{{elapsed}}{_BAR_RESET}, {_BAR_YELLOW}{{rate_fmt}}{_BAR_RESET}{{postfix}}]"
)


def capped_bar_cols(width: int | None) -> int | None:
    """The bar width to use for a console `width` columns wide (None = unknown, left to tqdm)."""
    if width is None:
        return None
    return max(1, min(MAX_BAR_COLS, width))


class ProgressBar(tqdm):
    """tqdm with the same arguments and behavior, but at most MAX_BAR_COLS wide, with coloured stats."""

    def __init__(self, *args, **kwargs):
        self._style_bar_stats = "bar_format" not in kwargs
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        if "ncols" in d:
            d["ncols"] = self.ncols = capped_bar_cols(d["ncols"])
        if getattr(self, "_style_bar_stats", False):
            d["bar_format"] = BAR_FORMAT if d.get("total") else BAR_FORMAT_NO_TOTAL
        return d


# Pipelined Hydrus calls for "Finding potential duplicates" (see _find_potential_duplicates_pipelined).
PIPELINE_DEPTH = 8  # at most this many searched files wait for their Hydrus calls


def pipeline_enabled() -> bool:
    """Set the environment variable HVD_PIPELINE=0 to make the Hydrus calls one after another on the main thread."""
    return os.environ.get("HVD_PIPELINE", "1").strip() != "0"


def skip_repeat_pairs_enabled_by_env() -> bool:
    """Set the environment variable HVD_SKIP_REPEAT_PAIRS=1 to turn on --skip-repeat-pairs (off by default)."""
    return os.environ.get("HVD_SKIP_REPEAT_PAIRS", "0").strip().lower() in ("1", "true", "yes", "on")


# Printed at the start of "Finding potential duplicates" when --skip-repeat-pairs is on. Display only.
SKIP_REPEAT_PAIRS_INFO = "Skipping repeat pair sends."

# Printed once at startup after the database filesize (or "creating a database") lines,
# for everyone (display only). Only the word "WARNING:" is red3 (not bold);
# --clear-search-cache is magenta; the rest is plain.
# One paragraph so it wraps to the console width (fewer lines on a typical 1080p window).
SEARCH_CACHE_WARNING = (
    "WARNING: If you clear or resync potential duplicates in Hydrus "
    '("delete all potential duplicate pairs and re-search" '
    'or "resync potential duplicate pairs to storage"), you will have to clear this tool\'s search cache '
    "(--clear-search-cache) and re-search every video from zero. A large library can take several hours or days."
)


GPU_DECODE_ON = "Using GPU decode (CUDA)."
GPU_DECODE_NO_DEVICE = "WARNING: --gpu-decode: no CUDA device found, decoding on the CPU."
GPU_DECODE_AUTO_ON = "Using GPU decode (auto)."
GPU_DECODE_AUTO_NO_DEVICE = "WARNING: --gpu-decode auto: no CUDA device found, decoding on the CPU."
GPU_DECODE_NO_NVDEC = (
    "WARNING: GPU decode is off: your NVIDIA driver is missing video decoding support.\n"
    "Decoding on the CPU instead. Fix: update your NVIDIA driver, or see the README."
)

# print_gpu_decode_status() results
GPU_STATUS_ON = "on"
GPU_STATUS_NO_DEVICE = "no_device"
GPU_STATUS_NO_NVDEC = "no_nvdec"


def print_gpu_decode_status(auto: bool = False) -> str:
    """Say once whether GPU decoding can be used, with a blank line after. Returns a GPU_STATUS_* value.

    No CUDA device: plain --gpu-decode keeps trying per video (unchanged), auto doesn't start its GPU worker.
    CUDA but no NVIDIA video decoder library (NVDEC): GPU decode is turned off for the run.
    """
    try:
        from .vpdqpy.vpdqpy import Vpdq

        status = GPU_STATUS_NO_DEVICE
        if Vpdq.cuda_decode_available():
            status = GPU_STATUS_ON if Vpdq.nvdec_library_available() else GPU_STATUS_NO_NVDEC
    except Exception:
        status = GPU_STATUS_NO_DEVICE
    if status == GPU_STATUS_ON:
        print(f"[blue]{GPU_DECODE_AUTO_ON if auto else GPU_DECODE_ON}")
    elif status == GPU_STATUS_NO_NVDEC:
        console_print(style_user_text(GPU_DECODE_NO_NVDEC), highlight=False)
    else:
        console_print(style_user_text(GPU_DECODE_AUTO_NO_DEVICE if auto else GPU_DECODE_NO_DEVICE), highlight=False)
    print()
    return status


def gpu_decode_auto_summary(gpu_count: int, cpu_count: int) -> str:
    return f"GPU decode (auto): {gpu_count} {_plural(gpu_count, 'video')} decoded on the GPU, {cpu_count} on the CPU."


# --gpu-decode auto: how many downloaded videos may wait for a free worker (on top of the two being hashed).
AUTO_READY_FILES = 2
# --gpu-decode auto, low RAM: before each download, keep at least AUTO_RAM_RESERVE free after room for two copies of
# a video (the download briefly needs two). The video's size is estimated from the largest one so far, at least
# AUTO_RAM_MIN_FILE. With AUTO_RAM_HEADROOM more free, AUTO_READY_FILES may wait; with less, 1; below the reserve, the
# next video is downloaded only when nothing else is in memory. Videos above AUTO_RAM_ALONE_FRACTION of all RAM are
# hashed alone. Never blocks forever: with nothing in memory a download always starts.
GIB = 1024**3
AUTO_RAM_RESERVE = 2 * GIB
AUTO_RAM_HEADROOM = 2 * GIB
AUTO_RAM_MIN_FILE = 512 * 1024**2
AUTO_RAM_ALONE_FRACTION = 0.25
# --gpu-decode auto: how long to wait for the workers to finish their current video after the run (or the GUI's
# skip), and at exit after CTRL+C (the CTRL+C itself never waits).
AUTO_JOIN_S = 30.0
AUTO_EXIT_JOIN_S = 10.0
# --gpu-decode auto: when the GPU worker is idle, no video is waiting for it and the CPU worker is busy, the GPU worker
# takes the oldest waiting CPU video (decoded on the GPU) instead of sitting idle. Hashes are identical either way.
# Only this direction: the GPU path barely uses the CPU, while a heavy GPU video (e.g. H.264 4K) on the CPU is
# several times slower and was measured to make the whole run slower.
AUTO_SHARE_WORK = True


def auto_ready_limit(free_bytes: int | None, largest_file: int) -> int:
    """--gpu-decode auto: how many downloaded videos may wait for a worker, given free RAM (None: unknown)."""
    if free_bytes is None:
        return AUTO_READY_FILES
    need = AUTO_RAM_RESERVE + 2 * max(largest_file, AUTO_RAM_MIN_FILE)
    if free_bytes >= need + AUTO_RAM_HEADROOM:
        return AUTO_READY_FILES
    if free_bytes >= need:
        return 1
    return 0


def hash_alone(file_size: int, total_bytes: int | None) -> bool:
    """--gpu-decode auto: True if a video is so big compared to the RAM that it's hashed with nothing else in memory."""
    return total_bytes is not None and file_size > total_bytes * AUTO_RAM_ALONE_FRACTION


def logical_cpu_count() -> int:
    """CPUs this process may use (respects affinity and containers where Python can tell)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def auto_worker_settings(logical_cpus: int, job_count: int) -> tuple[int, bool]:
    """--gpu-decode auto on weak CPUs: (hashing threads for the GPU worker, whether it helps with CPU videos).

    Below 4 CPUs the GPU worker hashes with 1 thread so it doesn't slow the CPU worker. At 2 or fewer it also
    doesn't take CPU videos.
    """
    gpu_threads = 1 if logical_cpus < 4 else job_count
    return gpu_threads, AUTO_SHARE_WORK and logical_cpus > 2


def _join_threads(threads, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


def print_search_cache_warning() -> None:
    """Display-only: the Hydrus clear/resync warning, with a blank line before and after."""
    print()
    console_print(style_user_text(SEARCH_CACHE_WARNING), highlight=False)
    print()


# Printed (normal text) once at the start of "Finding potential duplicates", for everyone. Display only.
SPEED_TIPS = (
    "Speed tips: The final phase of the tool is limited by Hydrus's database queues.\n"
    "The fewer jobs Hydrus has queued, the faster this phase runs:\n"
    " - Close any Hydrus duplicates pages (they recount pairs as they arrive).\n"
    " - Pause duplicate auto-resolution rules.\n"
    " - Pause Hydrus idle/background maintenance.\n"
    " - Pause active downloads and subscriptions."
)


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


def failed_page_sent_message(count: int, page_name: str) -> str:
    return f"Sent {count} failed {_plural(count, 'file')} to the Hydrus page '{page_name}'."


def failed_page_missing_tip(page_name: str) -> str:
    return (
        f"Tip: Create a Hydrus page with the name '{page_name}' before running the program "
        "to see any files that failed to hash."
    )


def found_videos_to_hash_message(n: int) -> str:
    if n == 0:
        return "No new videos to perceptually hash."
    return f"Found {n} {_plural(n, 'video')} to perceptually hash."


def added_hashes_message(n: int) -> str:
    return f"Added {n} new perceptual {_plural(n, 'hash', 'hashes')} to the database."


def finished_hashing_message(n: int) -> str:
    return f"Finished perceptual hashing. {added_hashes_message(n)}"


def duplicates_result_message(n: int) -> str:
    if n == 0:
        return "No new potential duplicate pairs found."
    return f"{n} new potential duplicate {_plural(n, 'pair')} marked for manual processing."


def finished_duplicates_message(n: int) -> str:
    if n == 0:
        return "Finished finding potential duplicates. No new potential duplicate pairs found."
    return f"Finished finding potential duplicates. {duplicates_result_message(n)}"


class RepeatPairSkipper:
    """
    Opt-in (--skip-repeat-pairs / HVD_SKIP_REPEAT_PAIRS=1): send each potential duplicate pair to Hydrus once per
    run instead of usually twice.

    The search finds most pairs from both sides (file A finds B, later B finds A) and the original code sends both.
    Hydrus stores a pair the same way whichever file comes first, and the second send just deletes and re-inserts
    the identical row, so skipping it leaves Hydrus with exactly the same pairs (in normal use, see the warning).

    * A pair is skipped only if that exact pair was already sent SUCCESSFULLY earlier in this run. A pair is
      recorded only after its Hydrus call returned without an error, so a failed or interrupted call is never
      treated as sent. Nothing is ever skipped because a file was searched in an earlier run.
    * Nothing is saved: the record lives in memory for this run only, and the tool's database is untouched.
    * Once the repeat is skipped the pair is forgotten again, so the memory used stays small.
    * The calls that are made are the original calls, in the original order, just without the repeats.
    * Skipped pairs still count as found, so the printed totals are unchanged.
    """

    def __init__(self, send: Callable[[str, str], None]):
        self._send = send
        self._sent: set = set()
        self.skipped = 0

    @staticmethod
    def _key(hash_a: str, hash_b: str):
        """The same key for (A, B) and (B, A)."""
        lo, hi = (hash_a, hash_b) if hash_a <= hash_b else (hash_b, hash_a)
        if len(lo) == 64 and len(hi) == 64:
            try:
                key = bytes.fromhex(lo + hi)  # compact: 64 bytes instead of two 64 character strings
                if len(key) == 64:
                    return key
            except ValueError:
                pass
        return (lo, hi)

    def __call__(self, hash_a: str, hash_b: str) -> None:
        key = self._key(hash_a, hash_b)
        if key in self._sent:
            self._sent.discard(key)
            self.skipped += 1
            return
        self._send(hash_a, hash_b)  # if this raises, the pair is NOT recorded as sent
        self._sent.add(key)


class HydrusApiException(Exception):
    """Wrapper around hydrus_api.HydrusAPIException to avoid some coupling with hydrus_api outside of FileHasher."""


class FailedPerceptualHashException(Exception):
    """Exception for when files are failed to be perceptually hashed."""

    def __init__(self, other_exc: str = ""):
        super().__init__(other_exc)
        self.other_exc = other_exc


class FileHasher:
    """
    A class to fetch a file from Hydrus and phash it.
    """

    def __init__(self, client: HVDClient, num_threads: int = 0, gpu_decode: bool = False):
        self.client = client
        self.num_threads = num_threads
        self.gpu_decode = gpu_decode

    def _fetch_file(self, file_hash: str):
        try:
            response = self.client.client.get_file(hash_=file_hash)
        except hydrus_api.HydrusAPIException as exc:
            raise HydrusApiException(exc)
        return response.content

    def _phash_file(self, file: bytes, gpu_decode: bool | None = None) -> bytes:
        if gpu_decode is None:
            gpu_decode = self.gpu_decode
        try:
            phash = compute_phash(file, self.num_threads, gpu_decode)
            phash_bytes: bytes = phash.bytes
        except Exception as exc:
            raise FailedPerceptualHashException(str(exc))

        # sanity check
        # Note: Hashes may have 0 bytes if there was no frames that were high enough quality to be used.
        if phash_bytes is None or (len(phash_bytes) % vpdq.VpdqHash.bytesPerPdqHash != 0):
            raise FailedPerceptualHashException("phash_str was None or len not multiple of 32.")

        return phash_bytes

    def fetch_file_or_error(self, file_hash: str) -> bytes | HydrusApiException:
        """
        Retrieves the file from Hydrus. Returns the file bytes, or the HydrusApiException if it failed.

        Safe to run in a background thread so the next file downloads while the current one is hashed.
        """
        try:
            return self._fetch_file(file_hash)
        except HydrusApiException as exc:
            # Add a delay before turning so that if there is some transient issue
            # the next file to be hashed won't also immediately error.
            # This is a hack. There should probably be some counter in the result generator
            # for the number of failures before hashing is stopped entirely.
            time.sleep(3)
            return exc

    def phash_fetched_file(
        self, file_hash: str, file_or_error: bytes | HydrusApiException, gpu_decode: bool | None = None
    ) -> PerceptuallyHashedFile | FailedPerceptuallyHashedFile:
        """Calculates the perceptual hash of an already fetched file and returns the result.

        gpu_decode overrides this FileHasher's setting for this one file (used by --gpu-decode auto).
        """
        if isinstance(file_or_error, HydrusApiException):
            return FailedPerceptuallyHashedFile(file_hash, file_or_error)

        try:
            phash = self._phash_file(file_or_error, gpu_decode)
        except FailedPerceptualHashException as exc:
            return FailedPerceptuallyHashedFile(file_hash, exc)

        return PerceptuallyHashedFile(file_hash, phash)

    def fetch_and_phash_file(self, file_hash: str) -> PerceptuallyHashedFile | FailedPerceptuallyHashedFile:
        """
        Retrieves the file from Hydrus and calculates its perceptual hash and returns the result.

        Returns FailedPerceptuallyHashedFile with the failed video hash if there are any errors.
        """
        return self.phash_fetched_file(file_hash, self.fetch_file_or_error(file_hash))


@dataclass
class PerceptualHashingStats:
    success_hash_count: int = 0
    failed_from_api_errors_count: int = 0
    failed_from_phash_count: int = 0
    # --gpu-decode auto only: videos decoded on each (failed hashes included, Hydrus download errors not).
    # A GPU video that fell back to CPU decoding counts as CPU.
    gpu_decoded_count: int = 0
    cpu_decoded_count: int = 0


class CancelledPerceptualHashException(Exception):
    """Exception for when perceptual hashing is cancelled."""

    def __init__(self, stats: PerceptualHashingStats):
        super().__init__()
        self.stats = stats


@dataclass
class NoneProgress:
    placeholder: None


@dataclass
class HashingProgress:
    complete: int
    total: int


@dataclass
class BuildingSearchTreeProgress:
    complete: int
    total: int


@dataclass
class SearchingForDuplicatesProgress:
    complete: int
    total: int


@dataclass
class DoneProgress:
    placeholder: None


@dataclass
class DedupeProgress:
    progress: NoneProgress | HashingProgress | BuildingSearchTreeProgress | SearchingForDuplicatesProgress


if TYPE_CHECKING:
    UpdateDedupeProgressCallback: TypeAlias = Callable[[DedupeProgress], None]


class HydrusVideoDeduplicator:
    hydlog = logging.getLogger("hvd")
    hydlog.setLevel(logging.INFO)
    _DEBUG = False
    skip_repeat_pairs = False  # see RepeatPairSkipper

    def __init__(
        self,
        db: DedupeDB.DedupeDb,
        client: HVDClient,
        similarity_threshold: float = 50.0,
        job_count: int = -2,
        failed_page_name: str | None = None,
        custom_query: Sequence[str] | None = None,
        update_progress_callback: UpdateDedupeProgressCallback | None = None,
        should_skip_step_fn: Callable[[], bool] | None = None,
        gpu_decode: bool | str = False,
        skip_repeat_pairs: bool = False,
    ):
        self.db = db
        # False: CPU decode. True: GPU decode (--gpu-decode). GPU_DECODE_AUTO: --gpu-decode auto.
        self.gpu_decode = gpu_decode
        self.gpu_decode_auto = gpu_decode == GPU_DECODE_AUTO
        self.gpu_decode_available = False
        if gpu_decode:
            status = print_gpu_decode_status(auto=self.gpu_decode_auto)
            self.gpu_decode_available = status == GPU_STATUS_ON
            if status == GPU_STATUS_NO_NVDEC and not self.gpu_decode_auto:
                self.gpu_decode = False  # no video could be GPU decoded
        self.skip_repeat_pairs = skip_repeat_pairs
        self.client = client
        self.similarity_threshold = similarity_threshold
        self.job_count = job_count
        self.page_logger = HydrusPageLogger(self.client, resolve_failed_page_name(failed_page_name))
        self.search_tags = self.get_search_tags(custom_query)
        self.update_progress_callback = update_progress_callback
        if self.update_progress_callback:
            self.update_progress_callback(NoneProgress(placeholder=None))
        self.should_skip_step_fn = should_skip_step_fn

    def get_search_tags(self, custom_query: Sequence[str] | None) -> list[str]:
        # system:filetype tags are really inconsistent
        search_tags = [
            "system:filetype=video, gif, apng",
            "system:has duration",
            "system:file service is not currently in trash",
        ]

        if custom_query is not None:
            # Remove whitespace and empty strings
            custom_query = [x for x in custom_query if x.strip()]
            if len(custom_query) > 0:
                search_tags.extend(custom_query)
                print(f"[blue]Custom Query: {custom_query}")
                print()
        return search_tags

    def deduplicate(
        self,
        skip_hashing: bool,
    ) -> int:
        """
        Run all deduplicate functions.

        Dedupe Algorithm:
        1. Perceptually hash the videos.
        2. Insert the perceptual hashes into the vptree
        3. Search for similar videos in the vptree.
        4. Mark the similar videos as potential duplicates in Hydrus.

        Returns the number of similar pairs found during searching.
        """
        num_similar_pairs = 0

        if skip_hashing:
            print("[blue]Skipping perceptual hashing")
        else:
            video_hashes = list(self.client.get_video_hashes(self.search_tags))
            video_hashes = self.filter_unhashed(video_hashes)
            print_success(found_videos_to_hash_message(len(video_hashes)))
            print_user("\nTip: You can skip perceptual hashing at any time by pressing CTRL+C.\n")
            self.hydlog.info("Starting perceptual hash processing")
            self.db.begin_transaction()
            with self.db.conn:
                stats = PerceptualHashingStats()
                hashing_finished = False
                try:
                    stats = self.add_perceptual_hashes_to_db(video_hashes)
                except CancelledPerceptualHashException as exc:
                    # Interrupted, but on purpose.
                    stats = exc.stats
                    print("[yellow]Perceptual hash processing was interrupted! Progress was saved.")
                else:
                    hashing_finished = True
                    print_success(finished_hashing_message(stats.success_hash_count))
                finally:
                    if self.update_progress_callback:
                        self.update_progress_callback(NoneProgress(None))
                    if self.gpu_decode_auto and (stats.gpu_decoded_count or stats.cpu_decoded_count):
                        print(f"[blue]{gpu_decode_auto_summary(stats.gpu_decoded_count, stats.cpu_decoded_count)}")
                    # Print some useful stats and info for users
                    total_failures = stats.failed_from_api_errors_count + stats.failed_from_phash_count
                    if total_failures > 0:
                        print(f"[yellow]Perceptual hash processing had {total_failures} total failed files.")

                        if stats.failed_from_api_errors_count > 0:
                            print(
                                f"[yellow]{stats.failed_from_api_errors_count} failures were due to API errors. Ensure Hydrus is running and accessible before trying again."  # noqa: E501
                            )

                        if stats.failed_from_phash_count > 0:
                            print(
                                f"[yellow]{stats.failed_from_phash_count} failures were from an error during perceptual hashing. Are the files corrupted?"  # noqa: E501
                            )
                            if self.page_logger.sent > 0:
                                print()
                                print_success(
                                    failed_page_sent_message(self.page_logger.sent, self.page_logger.page_name)
                                )
                            elif self.page_logger.page_missing:
                                print_user(f"\n{failed_page_missing_tip(self.page_logger.page_name)}")
                    if not hashing_finished:
                        print_success(added_hashes_message(stats.success_hash_count))

        # Insert the perceptual hashed files into the vptree.
        print_user("\nTip: You can skip building the search tree at any time by pressing CTRL+C.\n")
        self.db.begin_transaction()
        with self.db.conn:
            try:
                self.process_phashed_file_queue()
            except KeyboardInterrupt:
                print("[yellow]Building the search tree was interrupted! Progress was saved.")
            else:
                print_success("Finished building the search tree.")

        self.db.begin_transaction()
        with self.db.conn:
            try:
                self.run_maintenance()
            except KeyboardInterrupt:
                print("[yellow]Maintenance was interrupted!")
            else:
                print_success("Finished maintenance.")

        # Number of potential duplicates before adding more.
        # This is just to print info for the user.
        # Note: This will be inaccurate if the user searches for duplicates in the Hydrus client
        #       while this is running.
        pre_dedupe_count = self.client.get_potential_duplicate_count_hydrus()

        print_user("\nTip: You can skip finding potential duplicates at any time by pressing CTRL+C.\n")
        self.db.begin_transaction()
        with self.db.conn:
            search_finished = False
            try:
                num_similar_pairs = self.find_potential_duplicates()
            except KeyboardInterrupt:
                print("[yellow]Searching for duplicates was interrupted! Progress was saved.")
            else:
                search_finished = True

        # Statistics for user
        post_dedupe_count = self.client.get_potential_duplicate_count_hydrus()
        new_dedupes_count = post_dedupe_count - pre_dedupe_count
        if search_finished:
            print_success(finished_duplicates_message(new_dedupes_count))
        else:
            print_success(duplicates_result_message(new_dedupes_count))

        self.hydlog.info(f"{num_similar_pairs} similar file pairs found.")
        self.hydlog.info("Deduplication done.")
        if self.update_progress_callback:
            self.update_progress_callback(DoneProgress(None))

        return num_similar_pairs

    def filter_unhashed(self, file_hashes: list[FileHash]) -> list[FileHash]:
        """
        Get only the files that have not been perceptually hashed in the db from a list of files.
        """
        all_phashed_files = set(self.db.get_phashed_files())
        return [file_hash for file_hash in file_hashes if file_hash not in all_phashed_files]

    def add_perceptual_hashes_to_db(self, video_hashes: Sequence[str]) -> PerceptualHashingStats:
        """
        Retrieves the video from Hydrus,
        calculates the perceptual hash,
        and then add it to the database.
        """
        stats = PerceptualHashingStats()
        try:
            with ProgressBar(
                smoothing=TQDM_SMOOTHING,
                total=len(video_hashes),
                desc="Perceptually hashing videos",
                dynamic_ncols=True,
                unit="video",
                colour="GREEN",
            ) as pbar:
                if self.gpu_decode_auto:
                    self._phash_videos_auto(list(video_hashes), FileHasher(self.client, self.job_count), stats, pbar)
                    gc.collect()
                    return stats
                filehasher = FileHasher(self.client, self.job_count, self.gpu_decode)
                # Download the next file from Hydrus in a background thread while the current file is hashed.
                # Only one file is fetched ahead, and results are still processed and saved one at a time
                # in the original order, so the database writes are exactly the same as before.
                video_hashes = list(video_hashes)
                prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hvd-prefetch")
                next_fetch = (
                    prefetcher.submit(filehasher.fetch_file_or_error, video_hashes[0]) if video_hashes else None
                )
                try:
                    for index, video_hash in enumerate(video_hashes):
                        if self.update_progress_callback:
                            self.update_progress_callback(HashingProgress(complete=pbar.n + 1, total=pbar.total))
                        if self.should_skip_step_fn and self.should_skip_step_fn():
                            return stats

                        fetched = next_fetch.result()
                        next_fetch = None
                        if index + 1 < len(video_hashes):
                            next_fetch = prefetcher.submit(filehasher.fetch_file_or_error, video_hashes[index + 1])
                        result = filehasher.phash_fetched_file(video_hash, fetched)
                        del fetched
                        self._save_phash_result(result, stats, pbar)

                        # Collect garbage now to avoid huge memory usage from the video files and frames.
                        gc.collect()
                        pbar.update(1)
                        if self.update_progress_callback:
                            self.update_progress_callback(HashingProgress(complete=pbar.n, total=pbar.total))
                finally:
                    # Don't wait on a pending download if hashing was stopped early.
                    prefetcher.shutdown(wait=False, cancel_futures=True)
        except KeyboardInterrupt:
            raise CancelledPerceptualHashException(stats)
        gc.collect()
        return stats

    def _save_phash_result(
        self, result: PerceptuallyHashedFile | FailedPerceptuallyHashedFile, stats: PerceptualHashingStats, pbar
    ) -> None:
        """Record one hashing result: queue it in the DB (commit every 4), or report the failure. Main thread only."""
        if isinstance(result, FailedPerceptuallyHashedFile):
            # We only want to add the failure to the page if the file was the actual cause of failure.
            if isinstance(result.exc, HydrusApiException):
                stats.failed_from_api_errors_count += 1
                # Clear the bar first so these lines start on their own line, then redraw the bar below.
                # file=pbar.fp makes this work for the console and for tests that capture the bar.
                with pbar.external_write_mode(file=pbar.fp):
                    print("[red]Hydrus API error during perceptual hashing:")
                    console_print(Text(str(result.exc)), highlight=False)
            else:
                stats.failed_from_phash_count += 1
                with pbar.external_write_mode(file=pbar.fp):
                    print("[red]Failed to perceptually hash a file.")
                    print(f"[red]Failed file SHA256 hash: {result.file_hash}")
                    console_print(Text(str(result.exc)), highlight=False)
                if self.page_logger:
                    self.page_logger.add_failed_video(result.file_hash)
        else:
            self.db.add_to_phashed_files_queue(result.file_hash, result.perceptual_hash)
            stats.success_hash_count += 1
            # Commit periodically to save progress in case something fails.
            COMMIT_BATCH_SIZE = 4
            if stats.success_hash_count % COMMIT_BATCH_SIZE == 0:
                self.db.commit()

    def _phash_videos_auto(
        self, video_hashes: list[str], filehasher: FileHasher, stats: PerceptualHashingStats, pbar
    ) -> None:
        """
        --gpu-decode auto: a GPU worker and a CPU worker hash different videos at the same time.

        * One download thread fetches the videos from Hydrus in order, at most AUTO_READY_FILES ahead (fewer when RAM
          is low, see auto_ready_limit), and reads each video's codec and resolution from its header (no decoding) to
          pick GPU or CPU (prefers_gpu_decode).
        * Each worker takes the oldest waiting video meant for it. With AUTO_SHARE_WORK, an idle GPU worker also takes
          the oldest waiting CPU video while the CPU worker is busy. Hashes are identical on the GPU and the CPU, so
          this only changes speed. Fewer than 4 CPUs: see auto_worker_settings.
        * A video that fails on the GPU is retried on the CPU (compute_phash), same as plain --gpu-decode.
        * Results come back to this (main) thread, which alone does the DB writes, the commit every 4 videos, the
          failed-page and progress updates (_save_phash_result), same as the normal loop. The order can differ.
        * No CUDA device: everything is hashed by the CPU worker.
        * Ctrl+C or the GUI's skip: no more videos are handed out. Videos still being hashed are not saved and are
          hashed again next run, like an interrupted video in the normal loop. After the skip or a normal end this
          waits up to AUTO_JOIN_S for the threads to stop; after Ctrl+C it returns at once and waits at exit.
        """
        use_gpu = self.gpu_decode_available
        gpu_threads, share_work = auto_worker_settings(logical_cpu_count(), filehasher.num_threads)
        gpu_filehasher = FileHasher(self.client, gpu_threads)
        total = len(video_hashes)
        cond = threading.Condition()
        waiting: list = []  # (file_hash, file bytes or HydrusApiException, prefers GPU, hash alone)
        busy = {True: False, False: False}  # by worker: True = GPU worker, False = CPU worker
        memory = {"videos": 0, "alone": 0, "largest": 0}  # videos waiting or being hashed; how many to hash alone
        fetch_done = threading.Event()
        stop = threading.Event()
        results: queue.Queue = queue.Queue()
        failure: list[BaseException] = []

        def may_download() -> bool:
            if memory["videos"] == 0:
                return True  # never wait with nothing in memory
            if memory["alone"]:
                return False
            status = memory_status()
            return len(waiting) < auto_ready_limit(status[0] if status else None, memory["largest"])

        def fetch_all() -> None:
            try:
                for file_hash in video_hashes:
                    with cond:
                        while not stop.is_set() and not may_download():
                            cond.wait(0.5)
                    if stop.is_set():
                        return
                    fetched = filehasher.fetch_file_or_error(file_hash)
                    is_file = not isinstance(fetched, HydrusApiException)
                    wants_gpu = use_gpu and is_file and video_prefers_gpu_decode(fetched)
                    alone = False
                    if is_file:
                        status = memory_status()
                        alone = hash_alone(len(fetched), status[1] if status else None)
                    with cond:
                        if is_file:
                            memory["largest"] = max(memory["largest"], len(fetched))
                        memory["videos"] += 1
                        memory["alone"] += alone
                        waiting.append((file_hash, fetched, wants_gpu, alone))
                        cond.notify_all()
                    del fetched
            except BaseException as exc:
                failure.append(exc)
            finally:
                fetch_done.set()
                with cond:
                    cond.notify_all()

        def take(on_gpu: bool):
            with cond:
                while not stop.is_set():
                    index = next((i for i, item in enumerate(waiting) if item[2] == on_gpu), None)
                    if index is None and waiting and share_work and on_gpu and busy[False]:
                        index = 0
                    if index is not None and waiting[index][3] and busy[not on_gpu]:
                        index = None  # a video to hash alone waits until the other worker is done
                    if index is not None and not any(item[3] for item in waiting[:index]) and not hashing_alone():
                        busy[on_gpu] = True
                        item = waiting.pop(index)
                        cond.notify_all()
                        return item
                    if fetch_done.is_set() and not waiting:
                        return None
                    cond.wait(0.5)
                return None

        def hashing_alone() -> bool:
            # a video being hashed alone: "alone" counts it while it's waiting or being hashed
            return memory["alone"] > sum(1 for item in waiting if item[3])

        def work(on_gpu: bool) -> None:
            hasher = gpu_filehasher if on_gpu else filehasher
            try:
                while not stop.is_set():
                    item = take(on_gpu)
                    if item is None:
                        return
                    file_hash, fetched, _, alone = item
                    del item
                    result = hasher.phash_fetched_file(file_hash, fetched, gpu_decode=on_gpu)
                    decoded = not isinstance(fetched, HydrusApiException)
                    used_gpu = last_decode_used_gpu() if on_gpu else False
                    decoded_on_gpu = on_gpu if used_gpu is None else used_gpu
                    del fetched
                    with cond:
                        busy[on_gpu] = False
                        memory["videos"] -= 1
                        memory["alone"] -= alone
                        cond.notify_all()
                    results.put((result, decoded_on_gpu, decoded))
            except BaseException as exc:
                failure.append(exc)

        workers = [threading.Thread(target=work, args=(False,), name="hvd-hash-cpu", daemon=True)]
        if use_gpu:
            workers.append(threading.Thread(target=work, args=(True,), name="hvd-hash-gpu", daemon=True))
        threads = [threading.Thread(target=fetch_all, name="hvd-prefetch", daemon=True), *workers]
        for thread in threads:
            thread.start()
        interrupted = False
        try:
            for _ in range(total):
                if self.update_progress_callback:
                    self.update_progress_callback(HashingProgress(complete=pbar.n + 1, total=pbar.total))
                while True:
                    if self.should_skip_step_fn and self.should_skip_step_fn():
                        return
                    try:
                        result, on_gpu, decoded = results.get(timeout=0.2)
                        break
                    except queue.Empty:
                        if failure:
                            raise failure[0]
                        if not any(worker.is_alive() for worker in workers) and results.empty():
                            return  # only if a worker died without reporting why; nothing is left to save
                if decoded:
                    if on_gpu:
                        stats.gpu_decoded_count += 1
                    else:
                        stats.cpu_decoded_count += 1
                self._save_phash_result(result, stats, pbar)
                del result
                # Collect garbage now to avoid huge memory usage from the video files and frames.
                gc.collect()
                pbar.update(1)
                if self.update_progress_callback:
                    self.update_progress_callback(HashingProgress(complete=pbar.n, total=pbar.total))
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            stop.set()
            with cond:
                waiting.clear()
                cond.notify_all()
            if interrupted:
                atexit.register(_join_threads, threads, AUTO_EXIT_JOIN_S)
            else:
                _join_threads(threads, AUTO_JOIN_S)

    def _find_potential_duplicates_pipelined(self, tree, files, search_threshold, send=None) -> int:
        """
        Same work as the loop in find_potential_duplicates(), but while Hydrus handles one file's pairs the next
        files are already being searched.

        * The main thread searches files in the same order as the original loop and puts each file's pairs, in the
          same order, into a small FIFO queue (PIPELINE_DEPTH files).
        * ONE sender thread takes the files in order and makes exactly the same set_file_relationships calls
          (same client and session, same bodies, one call at a time, same order) via mark_videos_as_duplicates().
        * Only after ALL of a file's calls succeeded does the sender hand the file back. The main thread, which owns
          the SQLite connection (sqlite3 connections must stay on their own thread), then does that file's DB write
          (searched_distance), the commit every 64 files and the progress update, in order, exactly as before.
          So a file is never marked as searched before Hydrus has received all of its pairs.
        * Ctrl+C: the sender stops after the call in flight; files whose calls all finished are marked, later
          ones are not (they are searched and sent again next run, exactly like an interrupted file today).
        * A Hydrus error stops the sender; files before it are marked and the same exception is raised here.
        * Any other error, or the GUI's skip, first lets the queued files finish, like the original loop would have.
        * `send` makes one Hydrus call (mark_videos_as_duplicates, or the opt-in RepeatPairSkipper wrapping it).
          It only runs on the sender thread, so a pair counts as sent only after its call succeeded.
        """
        if send is None:
            send = self.mark_videos_as_duplicates
        num_similar_pairs = 0
        processed_count = 0
        COMMIT_BATCH_SIZE = 64
        todo: queue.Queue = queue.Queue(maxsize=PIPELINE_DEPTH)
        done: queue.Queue = queue.Queue()
        stop = threading.Event()
        failure: list[BaseException] = []
        END = None

        def sender():
            try:
                while True:
                    try:
                        item = todo.get(timeout=0.05)
                    except queue.Empty:
                        if stop.is_set():
                            return
                        continue
                    if item is END or stop.is_set():
                        return
                    _, pairs = item
                    for file_hash_a, file_hash_b in pairs:
                        if stop.is_set():
                            return
                        self.hydlog.info(f'Similar files found: "{file_hash_a}" and "{file_hash_b}"')
                        send(file_hash_a, file_hash_b)
                    done.put(item)
            except BaseException as exc:  # reported to (and raised by) the main thread
                failure.append(exc)

        with ProgressBar(
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            total=len(files),
            desc="Finding potential duplicates",
            unit="video",
            colour="GREEN",
        ) as pbar:

            def apply_done():
                # Main thread: the original loop's per-file DB writes, for files whose Hydrus calls all succeeded.
                nonlocal num_similar_pairs, processed_count
                while True:
                    try:
                        hash_id, pairs = done.get_nowait()
                    except queue.Empty:
                        return
                    num_similar_pairs += len(pairs)
                    self.db.execute(
                        "UPDATE shape_search_cache SET searched_distance = ? WHERE hash_id = ?;",
                        (search_threshold, hash_id),
                    )
                    processed_count += 1
                    if processed_count % COMMIT_BATCH_SIZE == 0:
                        self.db.commit()
                    pbar.update(1)
                    if self.update_progress_callback:
                        self.update_progress_callback(SearchingForDuplicatesProgress(complete=pbar.n, total=pbar.total))

            def put(item):
                # Short timeouts keep Ctrl+C responsive on Windows while the queue is full.
                while True:
                    apply_done()
                    if failure:
                        raise failure[0]
                    try:
                        todo.put(item, timeout=0.05)
                        return
                    except queue.Full:
                        pass

            def wait_for_sender():
                while thread.is_alive():
                    thread.join(0.05)
                    apply_done()
                apply_done()

            thread = threading.Thread(target=sender, name="hvd-hydrus-sender", daemon=True)
            thread.start()
            try:
                for hash_id in files:
                    if self.update_progress_callback:
                        self.update_progress_callback(SearchingForDuplicatesProgress(complete=pbar.n, total=pbar.total))
                    if self.should_skip_step_fn and self.should_skip_step_fn():
                        # Stop searching, but still send and report the pairs found so far.
                        break

                    hash_id = hash_id[0]
                    result = tree.search_file(hash_id, max_hamming_distance=search_threshold)
                    file_hash_a = self.db.get_file_hash(hash_id)
                    pairs = []
                    for similar_hash_id, distance in result:
                        file_hash_b = self.db.get_file_hash(similar_hash_id)
                        if hash_id != similar_hash_id:
                            pairs.append((file_hash_a, file_hash_b))
                    put((hash_id, pairs))
                put(END)
                wait_for_sender()
                if failure:
                    raise failure[0]
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    stop.set()  # finish only the call in flight
                else:
                    while thread.is_alive():  # let the queued files finish first, like the original loop
                        try:
                            todo.put(END, timeout=0.05)
                            break
                        except queue.Full:
                            apply_done()
                try:
                    wait_for_sender()
                except KeyboardInterrupt:  # a (second) Ctrl+C while waiting: stop waiting
                    stop.set()
                    apply_done()
                raise
        return num_similar_pairs // 2

    def mark_videos_as_duplicates(self, video1_hash: str, video2_hash: str):
        """Mark a pair of videos as duplicates in Hydrus."""
        new_relationship = {
            "hash_a": video1_hash,
            "hash_b": video2_hash,
            "relationship": int(hydrus_api.DuplicateStatus.POTENTIAL_DUPLICATES),
            "do_default_content_merge": True,
        }

        self.client.client.set_file_relationships([new_relationship])

    def process_phashed_file_queue(self):
        """
        Process the files in the phashed files queue.
        This inserts the queue entries into their respective tables and then inserts the file into the vptree.
        """
        results = self.db.execute("SELECT file_hash, phash FROM phashed_file_queue").fetchall()
        # Faster, identical tree inserts (HVD_FAST_INSERT=0 disables). See vptree.fast_tree_building.
        fast_building = vptree.fast_tree_building(self.db) if results else contextlib.nullcontext(False)
        with (
            fast_building as fast_insert,
            ProgressBar(
                smoothing=TQDM_SMOOTHING,
                total=len(results),
                desc="Building search tree",
                dynamic_ncols=True,
                unit="video",
                colour="GREEN",
            ) as pbar,
        ):
            if fast_insert:
                with pbar.external_write_mode(file=pbar.fp):
                    print("[blue]Using fast search tree building.")
            processed_count = 0
            for file_hash, perceptual_hash in results:
                if self.update_progress_callback:
                    self.update_progress_callback(BuildingSearchTreeProgress(complete=pbar.n, total=pbar.total))
                if self.should_skip_step_fn and self.should_skip_step_fn():
                    return

                self.db.add_file(file_hash)
                self.db.add_perceptual_hash(perceptual_hash)
                self.db.associate_file_with_perceptual_hash(file_hash, perceptual_hash)
                self.db.execute(
                    "DELETE FROM phashed_file_queue WHERE file_hash = :file_hash AND phash = :phash",
                    {"file_hash": file_hash, "phash": perceptual_hash},
                )
                processed_count += 1
                # Commit periodically to save progress in case something fails.
                COMMIT_BATCH_SIZE = 64
                if processed_count % COMMIT_BATCH_SIZE == 0:
                    self.db.commit()
                pbar.update(1)
                if self.update_progress_callback:
                    self.update_progress_callback(BuildingSearchTreeProgress(complete=pbar.n, total=pbar.total))
            if self.update_progress_callback:
                self.update_progress_callback(BuildingSearchTreeProgress(complete=len(results), total=len(results)))

    def run_maintenance(self):
        """Run maintenance, if needed."""
        # TODO: Currently broken. If it's cancelled while running it will report all the branches as broken.
        return
        tree = vptree.VpTreeManager(self.db)

        if tree.maintenance_due():
            # TODO: Do further testing on this.
            print("[blue]Running search tree maintenance...")
            tree.maintain_tree()

    def find_potential_duplicates(
        self,
    ) -> int:
        """
        Find potential duplicates in the database and mark them as such in Hydrus.

        Returns the number of similar file pairs found.
        """
        tree = vptree.VpTreeManager(self.db)
        search_threshold = vptree.fix_vpdq_similarity(self.similarity_threshold)

        files = self.db.execute(
            "SELECT hash_id FROM shape_search_cache WHERE searched_distance is NULL or searched_distance < :threshold",
            {"threshold": search_threshold},
        ).fetchall()

        # Display-only spacing: blank line between sections, no leading spaces on "Using ..." lines,
        # and the pipelined line is grouped with the other Using lines (print order only).
        need_gap = False

        def _section_gap():
            nonlocal need_gap
            if need_gap:
                print()
            need_gap = True

        if len(files) > 0:
            # "Speed tips:" is bold cyan; the rest of that line and the bullet lines are plain; display only
            print_user(SPEED_TIPS)
            need_gap = True

        # Speed up the search with an exact in-memory distance index (falls back to the original code if it
        # can't be built, or if HVD_FAST_SEARCH=0). The tree isn't modified during this phase.
        printed_using = False
        if len(files) > 0 and tree.enable_fast_search_index():
            _section_gap()
            printed_using = True
            print("[blue]Using fast search index.")
            # Skip the part of the tree walk every search takes identically (HVD_SCAN_SEARCH=0 disables).
            if tree.enable_fast_search_scan():
                print("[blue]Using fast search scan.")
        if len(files) > 0 and pipeline_enabled():
            if not printed_using:
                _section_gap()
                printed_using = True
            print("[blue]Using pipelined Hydrus calls.")

        send = self.mark_videos_as_duplicates
        skipper = None
        if self.skip_repeat_pairs or skip_repeat_pairs_enabled_by_env():
            # Opt-in, see RepeatPairSkipper. With it off, `send` is exactly the original call.
            skipper = RepeatPairSkipper(send)
            send = skipper
            if len(files) > 0:
                # Blue info line only when skip-repeat is on, grouped with the "Using ..." lines; display only
                if not printed_using:
                    _section_gap()
                console_print(Text(SKIP_REPEAT_PAIRS_INFO, style="blue"), highlight=False)
                need_gap = True

        _section_gap()  # blank line before the progress bar when any of the sections above printed
        try:
            if len(files) > 0 and pipeline_enabled():
                return self._find_potential_duplicates_pipelined(tree, files, search_threshold, send)
            return self._find_potential_duplicates_serial(tree, files, search_threshold, send)
        finally:
            if skipper is not None:
                print(
                    f"[blue]Skipped {skipper.skipped} repeat pair sends "
                    "(each of those pairs had already been sent to Hydrus in this run)."
                )

    def _find_potential_duplicates_serial(self, tree, files, search_threshold, send=None) -> int:
        """The original search loop: Hydrus calls one after another on this thread (used with HVD_PIPELINE=0)."""
        if send is None:
            send = self.mark_videos_as_duplicates
        num_similar_pairs = 0
        with ProgressBar(
            smoothing=TQDM_SMOOTHING,
            dynamic_ncols=True,
            total=len(files),
            desc="Finding potential duplicates",
            unit="video",
            colour="GREEN",
        ) as pbar:
            processed_count = 0
            for hash_id in files:
                if self.update_progress_callback:
                    self.update_progress_callback(SearchingForDuplicatesProgress(complete=pbar.n, total=pbar.total))
                if self.should_skip_step_fn and self.should_skip_step_fn():
                    # Stop searching, but still report the pairs found so far.
                    break

                hash_id = hash_id[0]
                result = tree.search_file(hash_id, max_hamming_distance=search_threshold)
                file_hash_a = self.db.get_file_hash(hash_id)
                for similar_hash_id, distance in result:
                    file_hash_b = self.db.get_file_hash(similar_hash_id)
                    if hash_id != similar_hash_id:
                        self.hydlog.info(f'Similar files found: "{file_hash_a}" and "{file_hash_b}"')
                        send(file_hash_a, file_hash_b)
                        num_similar_pairs += 1

                # TODO:
                # Do we need to add the below line here? See _PerceptualHashesSearchForPotentialDuplicates in Hydrus.
                # group_of_hash_ids = self._STL( self._Execute( 'SELECT hash_id FROM shape_search_cache WHERE searched_distance IS NULL or searched_distance < ?;', ( search_distance, ) ).fetchmany( 10 ) )   # noqa: E501
                # Update the search cache
                self.db.execute(
                    "UPDATE shape_search_cache SET searched_distance = ? WHERE hash_id = ?;",
                    (search_threshold, hash_id),
                )

                processed_count += 1
                # Commit periodically to save progress in case something fails.
                COMMIT_BATCH_SIZE = 64
                if processed_count % COMMIT_BATCH_SIZE == 0:
                    self.db.commit()

                pbar.update(1)
                if self.update_progress_callback:
                    self.update_progress_callback(SearchingForDuplicatesProgress(complete=pbar.n, total=pbar.total))
        return num_similar_pairs // 2
