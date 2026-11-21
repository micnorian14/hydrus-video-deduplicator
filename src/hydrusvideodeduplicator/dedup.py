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

import contextlib
import gc
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import hydrusvideodeduplicator.hydrus_api as hydrus_api

from .client import HVDClient
from .db import DedupeDB, vptree
from .hashing import compute_phash
from .page_logger import HydrusPageLogger
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
# the window on every refresh (dynamic_ncols). 100 fits a console on half of a 1080p screen, and keeps the bar
# short enough that resizing the window doesn't smear old bar lines as much.
MAX_BAR_COLS = 100


def capped_bar_cols(width: int | None) -> int | None:
    """The bar width to use for a console `width` columns wide (None = unknown, left to tqdm)."""
    if width is None:
        return None
    return max(1, min(MAX_BAR_COLS, width))


class ProgressBar(tqdm):
    """tqdm with the same arguments and behavior, but at most MAX_BAR_COLS wide."""

    @property
    def format_dict(self):
        d = super().format_dict
        if "ncols" in d:
            d["ncols"] = self.ncols = capped_bar_cols(d["ncols"])
        return d

# Pipelined Hydrus calls for "Finding potential duplicates" (see _find_potential_duplicates_pipelined).
PIPELINE_DEPTH = 8  # at most this many searched files wait for their Hydrus calls


def pipeline_enabled() -> bool:
    """Set the environment variable HVD_PIPELINE=0 to make the Hydrus calls one after another on the main thread."""
    return os.environ.get("HVD_PIPELINE", "1").strip() != "0"


def skip_repeat_pairs_enabled_by_env() -> bool:
    """Set the environment variable HVD_SKIP_REPEAT_PAIRS=1 to turn on --skip-repeat-pairs (off by default)."""
    return os.environ.get("HVD_SKIP_REPEAT_PAIRS", "0").strip().lower() in ("1", "true", "yes", "on")


# Printed (yellow) at the start of "Finding potential duplicates" when --skip-repeat-pairs is on.
# Lines are kept under 80 characters so they don't get re-wrapped on a standard console.
SKIP_REPEAT_PAIRS_WARNING = (
    "Skipping repeat pair sends (faster).\n"
    'WARNING: Do NOT use "delete all potential duplicate pairs and re-search" or\n'
    '"resync potential duplicate pairs to storage" in Hydrus while this runs.\n'
    "If you do, you will have to clear this tool's search cache\n"
    "(--clear-search-cache) and re-search every video from zero,\n"
    "which can take DAYS on a large library."
)

# Printed (normal text) once at the start of "Finding potential duplicates", for everyone. Display only.
SPEED_TIPS = (
    "Speed tips: Hydrus does the heavy lifting in this step. For best speed while this runs:\n"
    " - Close any Hydrus duplicates pages (they recount pairs as they arrive).\n"
    " - Pause duplicate auto-resolution rules.\n"
    " - Pause Hydrus idle/background maintenance.\n"
    " - Optional: pause active downloads and subscriptions."
)


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

    def _phash_file(self, file: bytes) -> bytes:
        try:
            phash = compute_phash(file, self.num_threads, self.gpu_decode)
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
        self, file_hash: str, file_or_error: bytes | HydrusApiException
    ) -> PerceptuallyHashedFile | FailedPerceptuallyHashedFile:
        """Calculates the perceptual hash of an already fetched file and returns the result."""
        if isinstance(file_or_error, HydrusApiException):
            return FailedPerceptuallyHashedFile(file_hash, file_or_error)

        try:
            phash = self._phash_file(file_or_error)
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
        gpu_decode: bool = False,
        skip_repeat_pairs: bool = False,
    ):
        self.db = db
        self.gpu_decode = gpu_decode
        self.skip_repeat_pairs = skip_repeat_pairs
        self.client = client
        self.similarity_threshold = similarity_threshold
        self.job_count = job_count
        self.page_logger = None if failed_page_name is None else HydrusPageLogger(self.client, failed_page_name)
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
            print(f"[blue]Found {len(video_hashes)} eligible files to perceptually hash.")
            print("[cyan]\nTip: You can skip perceptual hashing at any time by pressing CTRL+C.\n")
            self.hydlog.info("Starting perceptual hash processing")
            self.db.begin_transaction()
            with self.db.conn:
                stats = PerceptualHashingStats()
                try:
                    stats = self.add_perceptual_hashes_to_db(video_hashes)
                except CancelledPerceptualHashException as exc:
                    # Interrupted, but on purpose.
                    stats = exc.stats
                    print("[yellow]Perceptual hash processing was interrupted! Progress was saved.")
                else:
                    print("[green]Finished perceptual hash processing.")
                finally:
                    if self.update_progress_callback:
                        self.update_progress_callback(NoneProgress(None))
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
                            print(
                                "[cyan]\nTip: You could have seen which files failed directly in Hydrus by "
                                "creating a Hydrus page with the name 'failed' and "
                                "running the program with '--failed-page-name=failed'\n"
                            )
                    print(f"[green]Added {stats.success_hash_count} new perceptual hashes to the database.")

        # Insert the perceptual hashed files into the vptree.
        print("[cyan]\nTip: You can skip building the search tree at any time by pressing CTRL+C.\n")
        self.db.begin_transaction()
        with self.db.conn:
            try:
                self.process_phashed_file_queue()
            except KeyboardInterrupt:
                print("[yellow]Building the search tree was interrupted! Progress was saved.")
            else:
                print("[green]Finished fully building the search tree.")

        self.db.begin_transaction()
        with self.db.conn:
            try:
                self.run_maintenance()
            except KeyboardInterrupt:
                print("[yellow]Maintenance was interrupted!")
            else:
                print("[green]Finished maintenance.")

        # Number of potential duplicates before adding more.
        # This is just to print info for the user.
        # Note: This will be inaccurate if the user searches for duplicates in the Hydrus client
        #       while this is running.
        pre_dedupe_count = self.client.get_potential_duplicate_count_hydrus()

        print("[cyan]\nTip: You can skip finding potential duplicates at any time by pressing CTRL+C.\n")
        self.db.begin_transaction()
        with self.db.conn:
            try:
                num_similar_pairs = self.find_potential_duplicates()
            except KeyboardInterrupt:
                print("[yellow]Searching for duplicates was interrupted! Progress was saved.")

        # Statistics for user
        post_dedupe_count = self.client.get_potential_duplicate_count_hydrus()
        new_dedupes_count = post_dedupe_count - pre_dedupe_count
        if new_dedupes_count > 0:
            print(f"[green]{new_dedupes_count} new potential duplicate pairs marked for manual processing!")
        else:
            print("[green]No new potential duplicate pairs found.")

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
                desc="Perceptually hashing files",
                dynamic_ncols=True,
                unit="file",
                colour="GREEN",
            ) as pbar:
                filehasher = FileHasher(self.client, self.job_count, self.gpu_decode)
                successful_phashed_count = 0
                # Download the next file from Hydrus in a background thread while the current file is hashed.
                # Only one file is fetched ahead, and results are still processed and saved one at a time
                # in the original order, so the database writes are exactly the same as before.
                video_hashes = list(video_hashes)
                prefetcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hvd-prefetch")
                next_fetch = prefetcher.submit(filehasher.fetch_file_or_error, video_hashes[0]) if video_hashes else None
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
                        if isinstance(result, FailedPerceptuallyHashedFile):
                            # We only want to add the failure to the page if the file was the actual cause of failure.
                            if isinstance(result.exc, HydrusApiException):
                                stats.failed_from_api_errors_count += 1
                                print("[red]Hydrus API error during perceptual hashing:")
                                print(Text(str(result.exc), style="red"))
                            else:
                                stats.failed_from_phash_count += 1
                                print("[red]Failed to perceptually hash a file.")
                                print(f"[red]Failed file SHA256 hash: {result.file_hash}")
                                print(Text(str(result.exc), style="red"))
                                if self.page_logger:
                                    self.page_logger.add_failed_video(result.file_hash)
                        else:
                            self.db.add_to_phashed_files_queue(result.file_hash, result.perceptual_hash)
                            stats.success_hash_count += 1
                            successful_phashed_count += 1
                            # Commit periodically to save progress in case something fails.
                            COMMIT_BATCH_SIZE = 4
                            if successful_phashed_count % COMMIT_BATCH_SIZE == 0:
                                self.db.commit()

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
            dynamic_ncols=True, total=len(files), desc="Finding potential duplicates", unit="file", colour="GREEN"
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
        with fast_building as fast_insert, ProgressBar(
            smoothing=TQDM_SMOOTHING,
            total=len(results),
            desc="Building search tree",
            dynamic_ncols=True,
            unit="file",
            colour="GREEN",
        ) as pbar:
            if fast_insert:
                pbar.write("Using fast search tree building.")
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

        if len(files) > 0:
            # Header in cyan, tips in plain text (no auto-highlighting); display only
            tips_header, _, tips_rest = SPEED_TIPS.partition("\n")
            print(Text.assemble((tips_header, "cyan"), "\n", tips_rest))

        # Speed up the search with an exact in-memory distance index (falls back to the original code if it
        # can't be built, or if HVD_FAST_SEARCH=0). The tree isn't modified during this phase.
        if len(files) > 0 and tree.enable_fast_search_index():
            print("[blue] Using fast search index.")
            # Skip the part of the tree walk every search takes identically (HVD_SCAN_SEARCH=0 disables).
            if tree.enable_fast_search_scan():
                print("[blue] Using fast search scan.")

        send = self.mark_videos_as_duplicates
        skipper = None
        if self.skip_repeat_pairs or skip_repeat_pairs_enabled_by_env():
            # Opt-in, see RepeatPairSkipper. With it off, `send` is exactly the original call.
            skipper = RepeatPairSkipper(send)
            send = skipper
            # First line blue (info); in the warning only the word "WARNING:" is bold yellow; display only
            first_line, _, warning_lines = SKIP_REPEAT_PAIRS_WARNING.partition("\n")
            _, _, warning_rest = warning_lines.partition("WARNING:")
            print(Text.assemble((first_line, "blue"), "\n", ("WARNING:", "bold yellow"), warning_rest))
        try:
            if len(files) > 0 and pipeline_enabled():
                print("[blue] Using pipelined Hydrus calls.")
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
            dynamic_ncols=True, total=len(files), desc="Finding potential duplicates", unit="file", colour="GREEN"
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
