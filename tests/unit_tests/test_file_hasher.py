"""FileHasher fetch error handling: mid-download failures must not abort the hashing phase."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import requests

from hydrusvideodeduplicator import dedup

from .test_gpu_decode_auto import _FakeDb, _FakePhash
from .test_job_count import GIB, _Deduper


class _FailingBodyResponse:
    def __init__(self, exc: Exception):
        self._exc = exc
        self.closed = False

    def iter_content(self, chunk_size=1, decode_unicode=False):
        raise self._exc

    @property
    def content(self):
        raise self._exc

    def close(self):
        self.closed = True


class _StreamResponse:
    def __init__(self, chunks, delay=0.0, started=None):
        self.chunks = list(chunks)
        self.delay = delay
        self.started = started
        self.closed = False

    def iter_content(self, chunk_size=1, decode_unicode=False):
        if self.started is not None:
            self.started.set()
        for chunk in self.chunks:
            if self.delay:
                time.sleep(self.delay)
            yield chunk

    def close(self):
        self.closed = True


class TestFileHasherFetch(unittest.TestCase):
    def _hasher(self, response):
        client = mock.MagicMock()
        client.client.get_file.return_value = response
        return dedup.FileHasher(client)

    def test_fetch_file_chunked_encoding_error_raises_and_closes(self):
        response = _FailingBodyResponse(requests.exceptions.ChunkedEncodingError("truncated"))
        hasher = self._hasher(response)
        with self.assertRaises(dedup.HydrusApiException) as ctx:
            hasher._fetch_file("a" * 64)
        self.assertIsInstance(ctx.exception.args[0], requests.exceptions.ChunkedEncodingError)
        self.assertTrue(response.closed)

    def test_fetch_file_connection_error_raises_and_closes(self):
        response = _FailingBodyResponse(requests.exceptions.ConnectionError("dropped"))
        hasher = self._hasher(response)
        with self.assertRaises(dedup.HydrusApiException):
            hasher._fetch_file("a" * 64)
        self.assertTrue(response.closed)

    def test_fetch_file_or_error_returns_exception_for_body_read_failure(self):
        response = _FailingBodyResponse(requests.exceptions.ChunkedEncodingError("truncated"))
        hasher = self._hasher(response)
        with mock.patch.object(dedup.time, "sleep"):
            # The 3s retry delay is now an Event.wait loop, not time.sleep.
            with mock.patch.object(hasher, "_sleep_unless_cancelled", return_value=True):
                result = hasher.fetch_file_or_error("a" * 64)
        self.assertIsInstance(result, dedup.HydrusApiException)
        self.assertTrue(response.closed)

    def test_fetch_file_success_reads_content_and_closes(self):
        response = _StreamResponse([b"video-bytes"])
        hasher = self._hasher(response)
        self.assertEqual(hasher._fetch_file("a" * 64), b"video-bytes")
        self.assertTrue(response.closed)

    def test_fetch_file_joins_chunks_identically(self):
        chunks = [b"video-bytes-part1", b"part2", b"\x00\xff", b""]
        response = _StreamResponse(chunks)
        hasher = self._hasher(response)
        self.assertEqual(hasher._fetch_file("a" * 64), b"".join(chunks))
        self.assertTrue(response.closed)

    def test_read_response_body_empty_is_empty_bytes(self):
        self.assertEqual(dedup.read_response_body(_StreamResponse([])), b"")
        self.assertEqual(dedup.read_response_body(_StreamResponse([b"", b""])), b"")

    def test_cancel_aborts_slow_stream_and_closes_response(self):
        started = threading.Event()
        # 8 chunks at 0.25s would take 2.0s if the fetch ran to the end.
        response = _StreamResponse([b"x" * 1024] * 8, delay=0.25, started=started)
        hasher = self._hasher(response)

        def cancel_soon():
            self.assertTrue(started.wait(timeout=2))
            hasher.cancel_downloads()

        thread = threading.Thread(target=cancel_soon)
        started_at = time.perf_counter()
        thread.start()
        with self.assertRaises(dedup.DownloadCancelled):
            hasher._fetch_file("a" * 64)
        thread.join(timeout=2)
        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 1.0)
        self.assertTrue(response.closed)

    def test_fetch_file_or_error_reraises_cancel_not_a_failed_file(self):
        started = threading.Event()
        response = _StreamResponse([b"x" * 1024] * 8, delay=0.25, started=started)
        hasher = self._hasher(response)
        threading.Thread(target=lambda: started.wait(timeout=2) and hasher.cancel_downloads(), daemon=True).start()
        with self.assertRaises(dedup.DownloadCancelled):
            hasher.fetch_file_or_error("a" * 64)
        self.assertTrue(response.closed)

    def test_cancel_during_retry_delay_is_not_a_failed_file(self):
        response = _FailingBodyResponse(requests.exceptions.ConnectionError("dropped"))
        hasher = self._hasher(response)
        started = threading.Event()
        original = hasher._sleep_unless_cancelled

        def sleep_and_mark(seconds):
            started.set()
            return original(seconds)

        hasher._sleep_unless_cancelled = sleep_and_mark
        threading.Thread(
            target=lambda: started.wait(timeout=2) and hasher.cancel_downloads(),
            daemon=True,
        ).start()
        started_at = time.perf_counter()
        with self.assertRaises(dedup.DownloadCancelled):
            hasher.fetch_file_or_error("a" * 64)
        self.assertLess(time.perf_counter() - started_at, 1.0)
        self.assertTrue(response.closed)


class TestCancelledDownloadIsNotAFailedFile(unittest.TestCase):
    def test_skip_during_prefetch_does_not_record_failure(self):
        started_b = threading.Event()
        slow = _StreamResponse([b"y" * 1024] * 20, delay=0.2, started=started_b)
        client = mock.MagicMock()

        def get_file(hash_=None, **kwargs):
            if hash_ == "b":
                return slow
            return _StreamResponse([hash_.encode()])

        client.client.get_file.side_effect = get_file
        deduper = _Deduper.make(full_speed=True)
        deduper.client = client
        # Once the next video starts downloading, treat that as the user skipping.
        deduper.should_skip_step_fn = started_b.is_set
        deduper.page_logger = mock.MagicMock()

        def compute(video, num_threads, gpu_decode):
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "memory_status", lambda: (100 * GIB, 128 * GIB)),
        ):
            stats = deduper.add_perceptual_hashes_to_db(["a", "b", "c"])

        self.assertGreaterEqual(stats.success_hash_count, 1)
        self.assertIn("a", deduper.db.saved)
        self.assertNotIn("b", deduper.db.saved)
        self.assertNotIn("c", deduper.db.saved)
        self.assertEqual(stats.failed_from_api_errors_count, 0)
        self.assertEqual(stats.failed_from_phash_count, 0)
        deduper.page_logger.add_failed_video.assert_not_called()
        self.assertTrue(slow.closed)

    def test_auto_mode_skip_during_download_does_not_record_failure(self):
        started_b = threading.Event()
        slow = _StreamResponse([b"y" * 1024] * 20, delay=0.2, started=started_b)
        client = mock.MagicMock()

        def get_file(hash_=None, **kwargs):
            if hash_ == "b":
                return slow
            return _StreamResponse([hash_.encode()])

        client.client.get_file.side_effect = get_file
        deduper = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        deduper.db = _FakeDb()
        deduper.client = client
        deduper.job_count = 1
        deduper.gpu_decode = False
        deduper.gpu_decode_auto = True
        deduper.gpu_decode_available = False
        deduper.page_logger = mock.MagicMock()
        deduper.update_progress_callback = None
        deduper.should_skip_step_fn = started_b.is_set
        deduper.full_speed = True
        deduper.gpu_profile = None

        def compute(video, num_threads, gpu_decode):
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "video_prefers_gpu_decode", lambda _video: False),
            mock.patch.object(dedup, "memory_status", lambda: (100 * GIB, 128 * GIB)),
        ):
            stats = deduper.add_perceptual_hashes_to_db(["a", "b", "c"])

        self.assertNotIn("b", deduper.db.saved)
        self.assertEqual(stats.failed_from_api_errors_count, 0)
        self.assertEqual(stats.failed_from_phash_count, 0)
        deduper.page_logger.add_failed_video.assert_not_called()
        self.assertTrue(started_b.wait(timeout=2))
        self.assertTrue(slow.closed)


if __name__ == "__main__":
    unittest.main()
