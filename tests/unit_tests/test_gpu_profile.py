from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from hydrusvideodeduplicator import dedup, gpu_profile
from hydrusvideodeduplicator.gpu_profile import (
    CORRUPT,
    HARDWARE_CHANGED,
    LOADED,
    MIN_SAMPLES,
    NEW,
    RESET,
    DeviceStats,
    GpuProfile,
    bucket_key,
    resolution_bucket,
)

from .test_gpu_decode_auto import _FakeDb, _FakePhash

FP = {"cpu": "Test CPU", "logical_cpus": 12, "gpu": "Test GPU", "driver": "1.0", "gpu_device": "", "pyav": "19"}


def _profile(path=None, kinds=None) -> GpuProfile:
    """A profile with measured speeds: kinds = {(codec, w, h): (cpu cost or None, gpu cost or None)}."""
    profile = GpuProfile(path, FP)
    for (codec, w, h), costs in (kinds or {}).items():
        for on_gpu, cost in ((False, costs[0]), (True, costs[1])):
            if cost is not None:
                for _ in range(MIN_SAMPLES):
                    profile.record(codec, w, h, 1.0, on_gpu, cost)
    return profile


class TestBuckets(unittest.TestCase):
    def test_resolution_bucket(self):
        self.assertEqual(resolution_bucket(320, 240), "360p")
        self.assertEqual(resolution_bucket(1920, 1080), "1080p")
        self.assertEqual(resolution_bucket(1080, 1920), "1080p")  # portrait counts the same
        self.assertEqual(resolution_bucket(1920, 1088), "1080p")  # a little over still counts
        self.assertEqual(resolution_bucket(2560, 1440), "1440p")
        self.assertEqual(resolution_bucket(3840, 2160), "2160p")
        self.assertEqual(resolution_bucket(7680, 4320), "larger")
        self.assertEqual(resolution_bucket(0, 0), "unknown")
        self.assertEqual(resolution_bucket(None, None), "unknown")

    def test_bucket_key(self):
        self.assertEqual(bucket_key("H264", 1280, 720), "h264:720p")
        self.assertEqual(bucket_key("h265", 1920, 1080), "hevc:1080p")
        self.assertEqual(bucket_key(None, 1920, 1080), "unknown:1080p")


class TestRetuning(unittest.TestCase):
    def test_no_estimate_until_enough_samples(self):
        stats = DeviceStats()
        for _ in range(MIN_SAMPLES - 1):
            stats.add(1.0)
        self.assertFalse(stats.ready())
        stats.add(1.0)
        self.assertTrue(stats.ready())
        self.assertEqual(stats.estimate, 1.0)

    def test_first_estimate_is_the_median(self):
        stats = DeviceStats()
        for cost in [1.0, 1.0, 50.0, 1.0, 1.2]:  # one odd video
            stats.add(cost)
        self.assertEqual(stats.estimate, 1.0)

    def test_one_odd_video_does_not_move_it(self):
        stats = DeviceStats(1.0, 100)
        for cost in [1.0, 1.0, 1.0, 100.0, 1.0, 1.0]:
            stats.add(cost)
        self.assertAlmostEqual(stats.estimate, 1.0)

    def test_odd_stretch_moves_it_only_a_little(self):
        stats = DeviceStats(1.0, 100)
        for _ in range(3):
            stats.add(100.0)
        # 3 odd videos in a row: at most one step of EMA_WEIGHT toward MAX_STEP times the estimate
        self.assertLessEqual(stats.estimate, 1.0 + gpu_profile.EMA_WEIGHT * (gpu_profile.MAX_STEP - 1) + 1e-9)

    def test_lasting_change_is_followed(self):
        stats = DeviceStats(1.0, 100)
        for _ in range(40):
            stats.add(2.0)
        self.assertAlmostEqual(stats.estimate, 2.0, places=2)

    def test_record_ignores_unusable_timings(self):
        profile = GpuProfile(None, FP)
        self.assertFalse(profile.record("h264", 1920, 1080, None, False, 1.0))  # no duration
        self.assertFalse(profile.record("h264", 1920, 1080, 10.0, False, 0.0))
        self.assertFalse(profile.record("h264", 1920, 1080, float("nan"), False, 1.0))
        self.assertTrue(profile.record("h264", 1920, 1080, 10.0, False, 1.0))
        self.assertEqual(profile.samples("h264", 1920, 1080, on_gpu=False), 1)

    def test_time_is_per_second_of_video(self):
        profile = GpuProfile(None, FP)
        for _ in range(MIN_SAMPLES):
            profile.record("h264", 1920, 1080, 20.0, False, 2.0)
        self.assertAlmostEqual(profile.estimates("h264", 1920, 1080)[0], 0.1)
        for _ in range(MIN_SAMPLES):
            profile.record("h264", 1280, 720, 0.1, False, 0.25)  # very short clips count as MIN_DURATION_S
        self.assertAlmostEqual(profile.estimates("h264", 1280, 720)[0], 0.25 / gpu_profile.MIN_DURATION_S)


class TestRouting(unittest.TestCase):
    def test_fixed_rules_until_both_devices_measured(self):
        profile = _profile(kinds={("h264", 1920, 1080): (1.0, None), ("hevc", 1920, 1080): (None, 0.1)})
        self.assertFalse(profile.prefers_gpu("h264", 1920, 1080))  # fixed rule: 1080p H.264 on the CPU
        self.assertTrue(profile.prefers_gpu("hevc", 1920, 1080))  # fixed rule: HEVC on the GPU
        self.assertTrue(profile.prefers_gpu("h264", 3840, 2160))  # nothing measured: fixed rule

    def test_faster_device_once_measured(self):
        profile = _profile(kinds={("h264", 1920, 1080): (1.0, 0.5), ("hevc", 640, 360): (0.2, 0.4)})
        self.assertTrue(profile.prefers_gpu("h264", 1920, 1080))
        self.assertFalse(profile.prefers_gpu("hevc", 640, 360))

    def test_small_batch_keeps_fixed_rules(self):
        profile = GpuProfile(None, FP)
        for _ in range(MIN_SAMPLES - 1):
            profile.record("h264", 1920, 1080, 1.0, True, 0.01)  # GPU looks much faster, but too few videos
            profile.record("h264", 1920, 1080, 1.0, False, 10.0)
        self.assertFalse(profile.prefers_gpu("h264", 1920, 1080))

    def test_gpu_may_help(self):
        profile = _profile(kinds={("vp9", 1920, 1080): (1.0, 3.0), ("h264", 1280, 720): (1.0, 1.5)})
        self.assertFalse(profile.gpu_may_help(("vp9", 1920, 1080, 10.0)))  # 3x slower on the GPU
        self.assertTrue(profile.gpu_may_help(("h264", 1280, 720, 10.0)))
        self.assertTrue(profile.gpu_may_help(("av1", 1280, 720, 10.0)))  # not measured: help (and learn)
        self.assertTrue(profile.gpu_may_help(None))  # header unreadable: same as before the profiler

    def test_cpu_may_help(self):
        profile = _profile(kinds={("h264", 3840, 2160): (3.0, 1.0), ("hevc", 1920, 1080): (1.1, 1.0)})
        self.assertFalse(profile.cpu_may_help(("h264", 3840, 2160, 10.0), explore=True))  # much slower on the CPU
        self.assertTrue(profile.cpu_may_help(("hevc", 1920, 1080, 10.0), explore=False))  # about as fast
        self.assertFalse(profile.cpu_may_help(("hevc", 1280, 720, 10.0), explore=False))
        self.assertTrue(profile.cpu_may_help(("hevc", 1280, 720, 10.0), explore=True))  # not measured: explore
        self.assertFalse(profile.cpu_may_help(None, explore=True))


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / gpu_profile.PROFILE_FILE_NAME

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        profile = _profile(self.path, kinds={("h264", 1920, 1080): (1.0, 0.5), ("vp9", 1280, 720): (0.3, None)})
        self.assertTrue(profile.save())
        self.assertEqual([p.name for p in self.dir.iterdir()], [gpu_profile.PROFILE_FILE_NAME])  # no temp file left
        loaded, status = GpuProfile.load(self.path, FP)
        self.assertEqual(status, LOADED)
        self.assertEqual(loaded.estimates("h264", 1920, 1080), (1.0, 0.5))
        self.assertEqual(loaded.estimates("vp9", 1280, 720), (0.3, None))
        self.assertTrue(loaded.prefers_gpu("h264", 1920, 1080))

    def test_partial_samples_not_saved(self):
        profile = GpuProfile(self.path, FP)
        profile.record("h264", 1920, 1080, 1.0, False, 1.0)
        self.assertTrue(profile.save())
        self.assertEqual(json.loads(self.path.read_text())["buckets"], {})

    def test_nothing_learned_nothing_written(self):
        self.assertTrue(GpuProfile(self.path, FP).save())
        self.assertFalse(self.path.exists())

    def test_missing_file(self):
        profile, status = GpuProfile.load(self.path, FP)
        self.assertEqual(status, NEW)
        self.assertFalse(profile.prefers_gpu("h264", 1920, 1080))

    def test_corrupt_files(self):
        good = _profile(self.path, kinds={("h264", 1920, 1080): (1.0, 0.5)}).to_json()
        bad_estimate = json.loads(json.dumps(good))
        bad_estimate["buckets"]["h264:1080p"]["gpu"]["seconds_per_video_second"] = -1
        bad_samples = json.loads(json.dumps(good))
        bad_samples["buckets"]["h264:1080p"]["cpu"]["samples"] = "lots"
        for content in [
            "{not json",
            "",
            "[]",
            json.dumps({"version": 99, "fingerprint": FP, "buckets": {}}),
            json.dumps({"version": 1, "fingerprint": FP}),
            json.dumps(bad_estimate),
            json.dumps(bad_samples),
        ]:
            self.path.write_text(content, encoding="utf-8")
            profile, status = GpuProfile.load(self.path, FP)
            self.assertEqual(status, CORRUPT, content)
            self.assertFalse(profile.prefers_gpu("h264", 1920, 1080), content)  # fixed rules
        self.path.write_bytes(b"\xff\xfe\x00garbage")
        self.assertEqual(GpuProfile.load(self.path, FP)[1], CORRUPT)

    def test_corrupt_file_is_overwritten(self):
        self.path.write_text("{not json", encoding="utf-8")
        profile, _ = GpuProfile.load(self.path, FP)
        for _ in range(MIN_SAMPLES):
            profile.record("h264", 1920, 1080, 1.0, False, 1.0)
        self.assertTrue(profile.save())
        self.assertEqual(GpuProfile.load(self.path, FP)[1], LOADED)

    def test_hardware_change_starts_fresh(self):
        _profile(self.path, kinds={("h264", 1920, 1080): (1.0, 0.5)}).save()
        profile, status = GpuProfile.load(self.path, {**FP, "gpu": "Other GPU"})
        self.assertEqual(status, HARDWARE_CHANGED)
        self.assertEqual(profile.estimates("h264", 1920, 1080), (None, None))
        self.assertFalse(profile.prefers_gpu("h264", 1920, 1080))

    def test_reset_deletes_the_file(self):
        _profile(self.path, kinds={("h264", 1920, 1080): (1.0, 0.5)}).save()
        profile, status = GpuProfile.load(self.path, FP, reset=True)
        self.assertEqual(status, RESET)
        self.assertFalse(self.path.exists())
        self.assertEqual(profile.estimates("h264", 1920, 1080), (None, None))

    def test_failed_save_never_raises(self):
        profile = _profile(self.dir / "missing dir" / "gpu_profile.json", kinds={("h264", 1920, 1080): (1.0, 0.5)})
        self.assertFalse(profile.save())
        self.assertTrue(profile.save_error)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_no_path(self):
        profile, status = GpuProfile.load(None, FP)
        self.assertEqual(status, NEW)
        profile.record("h264", 1920, 1080, 1.0, False, 1.0)
        self.assertTrue(profile.save())


class TestSetup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / gpu_profile.PROFILE_FILE_NAME
        patcher = mock.patch.object(gpu_profile, "hardware_fingerprint", lambda threads: dict(FP))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_corrupt_profile_warns_and_continues(self):
        self.path.write_text("{not json", encoding="utf-8")
        with mock.patch.object(dedup, "console_print") as console:
            profile = dedup.setup_gpu_profile(self.dir, 11, reset=False, active=True)
        self.assertIsInstance(profile, GpuProfile)
        self.assertIn("Couldn't read the GPU profile", str(console.call_args))

    def test_reset_flag_outside_auto_mode_still_deletes(self):
        self.path.write_text("{}", encoding="utf-8")
        self.assertIsNone(dedup.setup_gpu_profile(self.dir, 11, reset=True, active=False))
        self.assertFalse(self.path.exists())

    def test_bad_db_dir_never_raises(self):
        self.assertIsInstance(dedup.setup_gpu_profile(mock.MagicMock(), 11, reset=False, active=True), GpuProfile)


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_without_nvidia_smi(self):
        with mock.patch.object(gpu_profile.subprocess, "run", side_effect=FileNotFoundError):
            fingerprint = gpu_profile.hardware_fingerprint(4)
        self.assertEqual((fingerprint["gpu"], fingerprint["driver"], fingerprint["hash_threads"]), ("", "", 4))
        self.assertTrue(fingerprint["logical_cpus"] >= 1)
        json.dumps(fingerprint)

    def test_fingerprint_reads_gpu(self):
        result = mock.Mock(returncode=0, stdout="NVIDIA GeForce RTX 5070 Ti, 581.57\n")
        with mock.patch.object(gpu_profile.subprocess, "run", return_value=result):
            fingerprint = gpu_profile.hardware_fingerprint(4)
        self.assertEqual((fingerprint["gpu"], fingerprint["driver"]), ("NVIDIA GeForce RTX 5070 Ti", "581.57"))


class TestAutoPipelineWithProfile(unittest.TestCase):
    """_phash_videos_auto with a GPU profile, fake downloads, a fake hasher and fake video headers."""

    def _deduper(self, profile):
        deduper = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        deduper.db = _FakeDb()
        deduper.client = None
        deduper.job_count = 1
        deduper.gpu_decode = "auto"
        deduper.gpu_decode_auto = True
        deduper.gpu_decode_available = True
        deduper.gpu_profile = profile
        deduper.page_logger = None
        deduper.update_progress_callback = None
        deduper.should_skip_step_fn = None
        return deduper

    # names: "<codec>-<height>-<n>" -> a 10 second video of that codec and 16:9 resolution
    @staticmethod
    def _probe(video: bytes):
        codec, height, _ = video.decode().split("-")
        height = int(height)
        return (codec, height * 16 // 9, height, 10.0)

    def _run(self, deduper, hashes, share_work=True, delay=0.0):
        decoded_on = {}
        lock = threading.Lock()

        def compute(video, num_threads, gpu_decode):
            with lock:
                decoded_on[video.decode()] = gpu_decode
            if delay:
                threading.Event().wait(delay)
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup.FileHasher, "_fetch_file", lambda _self, h: h.encode()),
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "probe_video_details", self._probe),
            mock.patch.object(dedup, "AUTO_SHARE_WORK", share_work),
            mock.patch.object(dedup, "logical_cpu_count", lambda: 12),
        ):
            stats = deduper.add_perceptual_hashes_to_db(hashes)
        return stats, decoded_on

    def test_measured_faster_gpu_gets_the_videos(self):
        profile = _profile(kinds={("h264", 1920, 1080): (1.0, 0.2)})
        hashes = [f"h264-1080-{i}" for i in range(6)]
        stats, decoded_on = self._run(self._deduper(profile), hashes, share_work=False)
        self.assertEqual(stats.success_hash_count, 6)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (6, 0))  # fixed rule would say CPU

    def test_learns_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gpu_profile.PROFILE_FILE_NAME
            profile = GpuProfile(path, FP)
            hashes = [f"h264-720-{i}" for i in range(8)]
            stats, _ = self._run(self._deduper(profile), hashes, share_work=False)
            self.assertEqual(stats.success_hash_count, 8)
            self.assertEqual(profile.samples("h264", 1280, 720, on_gpu=False), 8)
            self.assertTrue(path.exists())  # saved at the end of the run
            self.assertIn("h264:720p", json.loads(path.read_text())["buckets"])

    def test_cpu_never_takes_measured_heavy_gpu_videos(self):
        profile = _profile(kinds={("h264", 3840, 2160): (3.0, 1.0)})
        hashes = [f"h264-2160-{i}" for i in range(8)]
        stats, _ = self._run(self._deduper(profile), hashes, delay=0.03)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (8, 0))

    def test_idle_cpu_helps_when_about_as_fast(self):
        profile = _profile(kinds={("hevc", 1920, 1080): (1.05, 1.0)})
        hashes = [f"hevc-1080-{i}" for i in range(10)]
        stats, _ = self._run(self._deduper(profile), hashes, delay=0.03)
        self.assertEqual(stats.success_hash_count, 10)
        self.assertGreater(stats.cpu_decoded_count, 0)
        self.assertGreater(stats.gpu_decoded_count, 0)

    def test_gpu_skips_cpu_videos_measured_much_slower(self):
        profile = _profile(kinds={("vp9", 1920, 1080): (1.0, 5.0)})
        hashes = [f"vp9-1080-{i}" for i in range(8)]
        stats, _ = self._run(self._deduper(profile), hashes, delay=0.03)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (0, 8))

    def test_unreadable_header_uses_fixed_rules(self):
        profile = GpuProfile(None, FP)
        hashes = ["gpu0", "cpu0", "gpu1"]
        with (
            mock.patch.object(dedup, "video_prefers_gpu_decode", lambda video: video.startswith(b"gpu")),
            mock.patch.object(dedup.FileHasher, "_fetch_file", lambda _self, h: h.encode()),
            mock.patch.object(dedup, "compute_phash", lambda v, n, g: _FakePhash(v.ljust(32, b"\0")[:32])),
            mock.patch.object(dedup, "probe_video_details", lambda video: None),
            mock.patch.object(dedup, "AUTO_SHARE_WORK", False),
        ):
            stats = self._deduper(profile).add_perceptual_hashes_to_db(hashes)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (2, 1))


if __name__ == "__main__":
    unittest.main()
