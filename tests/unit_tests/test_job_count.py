"""--job-count caps decoding and hashing threads, the default and --full-speed modes, the RAM guard on the normal
hashing path, and MemoryError turning into one failed file."""

from __future__ import annotations

import io
import threading
import unittest
from unittest import mock

import av

from hydrusvideodeduplicator import config as hvd_config
from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.vpdqpy import vpdqpy
from hydrusvideodeduplicator.vpdqpy.vpdqpy import Vpdq

from .test_gpu_decode_auto import _FakeDb, _FakePhash

GIB = 1024**3


class TestJobCount(unittest.TestCase):
    def test_decode_thread_count_matches_hvdaccelerators_rules(self):
        self.assertEqual(vpdqpy.decode_thread_count(3, 12), 3)
        self.assertEqual(vpdqpy.decode_thread_count(0, 12), 0)  # FFmpeg picks: all threads, as before
        self.assertEqual(vpdqpy.decode_thread_count(-1, 12), 11)
        self.assertEqual(vpdqpy.decode_thread_count(-2, 12), 10)
        self.assertEqual(vpdqpy.decode_thread_count(-12, 12), 1)
        self.assertEqual(vpdqpy.decode_thread_count(-20, 12), 1)
        self.assertEqual(vpdqpy.decode_thread_count(-2, 0), 1)

    def test_resolve_job_count(self):
        self.assertEqual(hvd_config.resolve_job_count(4, 12), 4)
        self.assertEqual(hvd_config.resolve_job_count(0, 12), 12)
        self.assertEqual(hvd_config.resolve_job_count(-1, 12), 11)
        self.assertEqual(hvd_config.resolve_job_count(-2, 12), 10)
        self.assertEqual(hvd_config.resolve_job_count(-2, 2), 1)
        self.assertEqual(hvd_config.resolve_job_count(-5, 2), 1)

    def test_default_is_half_the_threads(self):
        for cpus, expected in ((1, 1), (2, 1), (3, 1), (4, 2), (6, 3), (8, 4), (12, 6), (16, 8), (32, 16)):
            with self.subTest(cpus=cpus):
                self.assertEqual(hvd_config.default_job_count(False, cpus), expected)
                self.assertEqual(hvd_config.default_job_count(True, cpus), 0)  # --full-speed: all threads

    def test_explicit_job_count_wins_in_both_modes(self):
        for full_speed in (False, True):
            self.assertEqual(hvd_config.effective_job_count(3, full_speed, 12), 3)
            self.assertEqual(hvd_config.effective_job_count("-1", full_speed, 12), -1)
            self.assertEqual(hvd_config.effective_job_count(0, full_speed, 12), 0)
        self.assertEqual(hvd_config.effective_job_count(None, False, 12), 6)
        self.assertEqual(hvd_config.effective_job_count(None, True, 12), 0)

    def test_config_job_count_unset_or_from_env(self):
        self.assertIsNone(hvd_config.Config._load({}).job_count)
        self.assertEqual(hvd_config.Config._load({"JOB_COUNT": "4"}).job_count, "4")

    def test_hash_passes_the_job_count_to_decoding_and_hashing(self):
        calls = {}

        class FakeHasher:
            def __init__(self, fps, width, height, threads):
                calls["hash_threads"] = threads

            def finish(self):
                return "hash"

        def extract(video, gpu_decode=False, decode_threads=0):
            calls["decode_threads"] = decode_threads
            return iter(())

        with (
            mock.patch.object(vpdqpy.vpdq, "VideoHasher", FakeHasher),
            mock.patch.object(Vpdq, "frame_extract_pyav", staticmethod(extract)),
            mock.patch.object(vpdqpy.os, "cpu_count", lambda: 12),
        ):
            for job_count, decode in ((1, 1), (6, 6), (0, 0), (-2, 10)):
                Vpdq._hash_video_bytes(b"video", job_count, gpu_decode=False)
                self.assertEqual(calls, {"hash_threads": job_count, "decode_threads": decode})


def _tiny_video() -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(25):
            frame = av.VideoFrame(64, 48, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(((i * 7 + j) % 256 for j in range(plane.buffer_size))))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


class TestDecodeThreadsReal(unittest.TestCase):
    def test_thread_count_is_set_before_decoding_and_hashes_match(self):
        video = _tiny_video()
        seen = []
        original_open = vpdqpy.av.open

        def spy_open(*args, **kwargs):
            container = original_open(*args, **kwargs)
            seen.append(container.streams.video[0])
            return container

        hashes = set()
        with mock.patch.object(vpdqpy.av, "open", spy_open):
            for threads in (1, 2, 0, -1):
                seen.clear()
                hashes.add(Vpdq.computeHash(video, threads).bytes)
                expected = vpdqpy.decode_thread_count(threads)
                if expected:
                    self.assertEqual(seen[-1].thread_count, expected)
                else:  # 0: FFmpeg picks (all CPU threads + 1) when the decoder opens
                    self.assertGreaterEqual(seen[-1].thread_count, vpdqpy.os.cpu_count() or 1)
        self.assertEqual(len(hashes), 1)  # same hash for any thread count


class _Deduper:
    @staticmethod
    def make(full_speed=True):
        deduper = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        deduper.db = _FakeDb()
        deduper.client = None
        deduper.job_count = 1
        deduper.gpu_decode = False
        deduper.gpu_decode_auto = False
        deduper.gpu_decode_available = False
        deduper.page_logger = mock.MagicMock()
        deduper.update_progress_callback = None
        deduper.should_skip_step_fn = None
        deduper.full_speed = full_speed
        return deduper


class TestMemoryErrorIsOneFailedFile(unittest.TestCase):
    def test_fetch_memory_error_becomes_failed_hash(self):
        client = mock.MagicMock()
        response = mock.MagicMock()
        type(response).content = mock.PropertyMock(side_effect=MemoryError())
        client.client.get_file.return_value = response
        hasher = dedup.FileHasher(client)
        result = hasher.fetch_file_or_error("a" * 64)
        self.assertIsInstance(result, dedup.FailedPerceptualHashException)
        self.assertIn(dedup.OUT_OF_MEMORY_MESSAGE, str(result))
        response.close.assert_called_once()
        failed = hasher.phash_fetched_file("a" * 64, result)
        self.assertIsInstance(failed, dedup.FailedPerceptuallyHashedFile)

    def test_hash_memory_error_becomes_failed_hash(self):
        hasher = dedup.FileHasher(None)
        with mock.patch.object(dedup, "compute_phash", side_effect=MemoryError()):
            failed = hasher.phash_fetched_file("a" * 64, b"video")
        self.assertIsInstance(failed, dedup.FailedPerceptuallyHashedFile)
        self.assertIn(dedup.OUT_OF_MEMORY_MESSAGE, str(failed.exc))

    def _run(self, deduper, hashes, fetch_fails=(), hash_fails=()):
        def fetch(_self, file_hash):
            if file_hash in fetch_fails:
                raise dedup.FailedPerceptualHashException(dedup.failed_hash_message(MemoryError()))
            return file_hash.encode()

        def compute(video, num_threads, gpu_decode):
            if video.decode() in hash_fails:
                raise MemoryError()
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup.FileHasher, "_fetch_file", fetch),
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "video_prefers_gpu_decode", lambda video: False),
            mock.patch.object(dedup, "memory_status", lambda: (100 * GIB, 128 * GIB)),
        ):
            return deduper.add_perceptual_hashes_to_db(hashes)

    def test_normal_path_run_continues(self):
        deduper = _Deduper.make()
        stats = self._run(deduper, ["v0", "v1", "v2", "v3"], fetch_fails={"v1"}, hash_fails={"v2"})
        self.assertEqual(stats.success_hash_count, 2)
        self.assertEqual(stats.failed_from_phash_count, 2)
        self.assertEqual(stats.failed_from_api_errors_count, 0)
        added = [c.args[0] for c in deduper.page_logger.add_failed_video.call_args_list]
        self.assertEqual(added, ["v1", "v2"])  # both go to the failed page

    def test_auto_path_run_continues(self):
        deduper = _Deduper.make()
        deduper.gpu_decode = "auto"
        deduper.gpu_decode_auto = True
        with mock.patch.object(dedup, "logical_cpu_count", lambda: 12):
            stats = self._run(deduper, ["v0", "v1", "v2", "v3"], fetch_fails={"v1"}, hash_fails={"v2"})
        self.assertEqual(stats.success_hash_count, 2)
        self.assertEqual(stats.failed_from_phash_count, 2)
        self.assertEqual(stats.cpu_decoded_count, 3)  # the out-of-memory download wasn't decoded
        added = sorted(c.args[0] for c in deduper.page_logger.add_failed_video.call_args_list)
        self.assertEqual(added, ["v1", "v2"])


class TestNormalPathRamGuard(unittest.TestCase):
    def test_may_prefetch(self):
        with mock.patch.object(dedup, "memory_status", lambda: None):
            self.assertTrue(dedup.may_prefetch(b"x", 0))
        with mock.patch.object(dedup, "memory_status", lambda: (100 * GIB, 128 * GIB)):
            self.assertTrue(dedup.may_prefetch(b"x", 1 * GIB))
            self.assertTrue(dedup.may_prefetch(b"x", 1 * GIB, full_speed=False))
        with mock.patch.object(dedup, "memory_status", lambda: (1 * GIB, 8 * GIB)):
            self.assertFalse(dedup.may_prefetch(b"x", 0))
        # Enough for one more video, not for the extra headroom the default (reduced) mode wants.
        with mock.patch.object(dedup, "memory_status", lambda: (4 * GIB, 64 * GIB)):
            self.assertTrue(dedup.may_prefetch(b"x", 0))
            self.assertFalse(dedup.may_prefetch(b"x", 0, full_speed=False))
        # A video bigger than a quarter of the RAM is hashed with nothing else downloading.
        with mock.patch.object(dedup, "hash_alone", lambda size, total: True):
            with mock.patch.object(dedup, "memory_status", lambda: (100 * GIB, 128 * GIB)):
                self.assertFalse(dedup.may_prefetch(b"x", 0))

    def test_auto_ready_files(self):
        self.assertEqual(dedup.auto_ready_files(True), dedup.AUTO_READY_FILES)
        self.assertEqual(dedup.auto_ready_files(False), 1)

    def _run(self, memory, full_speed=True):
        lock = threading.Lock()
        counts = {"fetched": 0, "computed": 0, "max_in_memory_at_fetch": 0}

        def fetch(_self, file_hash):
            with lock:
                counts["max_in_memory_at_fetch"] = max(
                    counts["max_in_memory_at_fetch"], counts["fetched"] - counts["computed"]
                )
                counts["fetched"] += 1
            return file_hash.encode()

        def compute(video, num_threads, gpu_decode):
            threading.Event().wait(0.02)
            with lock:
                counts["computed"] += 1
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup.FileHasher, "_fetch_file", fetch),
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "memory_status", lambda: memory),
        ):
            stats = _Deduper.make(full_speed).add_perceptual_hashes_to_db([f"v{i}" for i in range(6)])
        return stats, counts

    def test_low_ram_waits_until_the_current_video_is_hashed(self):
        stats, counts = self._run((1 * GIB, 8 * GIB))
        self.assertEqual(stats.success_hash_count, 6)  # never blocks forever
        self.assertEqual(counts["max_in_memory_at_fetch"], 0)

    def test_plenty_of_ram_downloads_ahead(self):
        stats, counts = self._run((100 * GIB, 128 * GIB))
        self.assertEqual(stats.success_hash_count, 6)
        self.assertEqual(counts["max_in_memory_at_fetch"], 1)

    def test_unknown_ram_downloads_ahead(self):
        stats, counts = self._run(None, full_speed=False)
        self.assertEqual(stats.success_hash_count, 6)
        self.assertEqual(counts["max_in_memory_at_fetch"], 1)


if __name__ == "__main__":
    unittest.main()
