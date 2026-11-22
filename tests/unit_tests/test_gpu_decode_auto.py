from __future__ import annotations

import threading
import unittest
from typing import Annotated, Optional
from unittest import mock

import typer
from typer.testing import CliRunner

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.config import (
    GPU_DECODE_AUTO,
    Config,
    normalize_gpu_decode_args,
    resolve_gpu_decode,
)
from hydrusvideodeduplicator.hashing import prefers_gpu_decode


class TestPrefersGpuDecode(unittest.TestCase):
    def test_hevc_any_resolution(self):
        for w, h in [(640, 360), (1920, 1080), (3840, 2160), (0, 0)]:
            self.assertTrue(prefers_gpu_decode("hevc", w, h))

    def test_h264_by_pixel_count(self):
        self.assertFalse(prefers_gpu_decode("h264", 888, 500))
        self.assertFalse(prefers_gpu_decode("h264", 1280, 720))
        self.assertFalse(prefers_gpu_decode("h264", 1920, 1080))
        self.assertFalse(prefers_gpu_decode("h264", 1080, 1920))  # portrait 1080p
        self.assertTrue(prefers_gpu_decode("h264", 2560, 1440))
        self.assertTrue(prefers_gpu_decode("h264", 2560, 1400))  # slightly cropped 1440p
        self.assertTrue(prefers_gpu_decode("h264", 1440, 2560))  # portrait 1440p
        self.assertTrue(prefers_gpu_decode("h264", 3840, 2160))
        self.assertTrue(prefers_gpu_decode("h264", 2160, 3840))  # portrait 4K
        self.assertTrue(prefers_gpu_decode("H264", 3840, 2160))

    def test_everything_else_on_cpu(self):
        for codec in ["vp8", "vp9", "mpeg4", "av1", "gif", "", None]:
            self.assertFalse(prefers_gpu_decode(codec, 3840, 2160), codec)
        self.assertFalse(prefers_gpu_decode("h264", None, None))


class TestGpuDecodeArgs(unittest.TestCase):
    def test_auto_value(self):
        self.assertEqual(normalize_gpu_decode_args(["--gpu-decode", "auto"]), ["--gpu-decode-auto"])
        self.assertEqual(normalize_gpu_decode_args(["--gpu-decode", "AUTO"]), ["--gpu-decode-auto"])
        self.assertEqual(normalize_gpu_decode_args(["--gpu-decode=auto"]), ["--gpu-decode-auto"])
        self.assertEqual(
            normalize_gpu_decode_args(["--api-url", "x", "--gpu-decode", "auto", "--verbose"]),
            ["--api-url", "x", "--gpu-decode-auto", "--verbose"],
        )

    def test_bare_flag_and_off_unchanged(self):
        self.assertEqual(normalize_gpu_decode_args(["--gpu-decode"]), ["--gpu-decode", "--no-gpu-decode-auto"])
        self.assertEqual(
            normalize_gpu_decode_args(["--gpu-decode", "--verbose"]),
            ["--gpu-decode", "--no-gpu-decode-auto", "--verbose"],
        )
        self.assertEqual(normalize_gpu_decode_args(["--no-gpu-decode"]), ["--no-gpu-decode", "--no-gpu-decode-auto"])
        self.assertEqual(normalize_gpu_decode_args([]), [])
        self.assertEqual(normalize_gpu_decode_args(["--verbose"]), ["--verbose"])
        self.assertEqual(normalize_gpu_decode_args(["--", "--gpu-decode", "auto"]), ["--", "--gpu-decode", "auto"])

    def test_resolve(self):
        self.assertEqual(resolve_gpu_decode(False, True), GPU_DECODE_AUTO)
        self.assertEqual(resolve_gpu_decode(True, True), GPU_DECODE_AUTO)
        self.assertIs(resolve_gpu_decode(True, False), True)
        self.assertIs(resolve_gpu_decode(False, False), False)
        self.assertIs(resolve_gpu_decode(None, None), False)

    def test_env_var(self):
        for value in ["1", "true", "yes", "on", " ON "]:
            config = Config._load({"GPU_DECODE": value})
            self.assertEqual((config.gpu_decode, config.gpu_decode_auto), (True, False), value)
        for value in ["auto", "AUTO", " auto "]:
            config = Config._load({"GPU_DECODE": value})
            self.assertEqual((config.gpu_decode, config.gpu_decode_auto), (False, True), value)
        for value in ["0", "false", "", "off"]:
            config = Config._load({"GPU_DECODE": value})
            self.assertEqual((config.gpu_decode, config.gpu_decode_auto), (False, False), value)
        config = Config._load({})
        self.assertEqual((config.gpu_decode, config.gpu_decode_auto), (False, False))

    def _parse(self, args: list[str], env_gpu_decode: bool = False, env_auto: bool = False):
        """Parse like entrypoint.main does: the same two options and the same argument normalization."""
        seen = {}

        def main(
            gpu_decode: Annotated[Optional[bool], typer.Option()] = env_gpu_decode,
            gpu_decode_auto: Annotated[Optional[bool], typer.Option(hidden=True)] = env_auto,
        ):
            seen["value"] = resolve_gpu_decode(gpu_decode, gpu_decode_auto)

        app = typer.Typer(add_completion=False)
        app.command()(main)
        result = CliRunner().invoke(app, normalize_gpu_decode_args(args))
        self.assertEqual(result.exit_code, 0, result.output)
        return seen["value"]

    def test_cli(self):
        self.assertIs(self._parse([]), False)
        self.assertIs(self._parse(["--gpu-decode"]), True)
        self.assertIs(self._parse(["--no-gpu-decode"]), False)
        self.assertEqual(self._parse(["--gpu-decode", "auto"]), GPU_DECODE_AUTO)
        self.assertEqual(self._parse(["--gpu-decode=auto"]), GPU_DECODE_AUTO)
        # GPU_DECODE=1 / GPU_DECODE=auto from the environment, and the command line overriding them
        self.assertIs(self._parse([], env_gpu_decode=True), True)
        self.assertEqual(self._parse([], env_auto=True), GPU_DECODE_AUTO)
        self.assertIs(self._parse(["--gpu-decode"], env_auto=True), True)
        self.assertIs(self._parse(["--no-gpu-decode"], env_auto=True), False)
        self.assertEqual(self._parse(["--gpu-decode", "auto"], env_gpu_decode=True), GPU_DECODE_AUTO)
        self.assertIs(self._parse(["--gpu-decode", "auto", "--gpu-decode"]), True)  # last one wins


class _FakePhash:
    def __init__(self, data: bytes):
        self.bytes = data


class _FakeDb:
    def __init__(self):
        self.saved: dict[str, bytes] = {}
        self.commits = 0
        self.threads: set[str] = set()

    def add_to_phashed_files_queue(self, file_hash, phash):
        self.threads.add(threading.current_thread().name)
        self.saved[file_hash] = phash

    def commit(self):
        self.threads.add(threading.current_thread().name)
        self.commits += 1


class TestAutoPipeline(unittest.TestCase):
    """_phash_videos_auto with fake downloads and a fake hasher (no video decoding, no Hydrus)."""

    def _deduper(self, gpu_available: bool):
        deduper = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        deduper.db = _FakeDb()
        deduper.client = None
        deduper.job_count = 1
        deduper.gpu_decode = GPU_DECODE_AUTO
        deduper.gpu_decode_auto = True
        deduper.gpu_decode_available = gpu_available
        deduper.page_logger = None
        deduper.update_progress_callback = None
        deduper.should_skip_step_fn = None
        return deduper

    def _run(self, deduper, hashes, share_work=False, delay=0.0, fail=()):
        decoded_on = {}

        def fetch(_self, file_hash):
            if file_hash.startswith("apierr"):
                raise dedup.HydrusApiException("boom")
            return file_hash.encode()

        def compute(video, num_threads, gpu_decode):
            name = video.decode()
            decoded_on[name] = gpu_decode
            if delay:
                threading.Event().wait(delay)  # time.sleep is patched out below (Hydrus retry delay)
            if name in fail:
                raise RuntimeError("bad video")
            return _FakePhash(name.encode().ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup.FileHasher, "_fetch_file", fetch),
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "video_prefers_gpu_decode", lambda video: video.startswith(b"gpu")),
            mock.patch.object(dedup, "AUTO_SHARE_WORK", share_work),
            mock.patch.object(dedup.time, "sleep", lambda s: None),
        ):
            stats = deduper.add_perceptual_hashes_to_db(hashes)
        return stats, decoded_on

    def test_routes_by_preference(self):
        hashes = [f"gpu{i}" for i in range(4)] + [f"cpu{i}" for i in range(6)]
        hashes = hashes[::2] + hashes[1::2]
        deduper = self._deduper(gpu_available=True)
        stats, decoded_on = self._run(deduper, hashes)
        self.assertEqual(stats.success_hash_count, 10)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (4, 6))
        self.assertEqual(decoded_on, {h: h.startswith("gpu") for h in hashes})
        self.assertEqual(set(deduper.db.saved), set(hashes))
        for h in hashes:
            self.assertEqual(deduper.db.saved[h], h.encode().ljust(32, b"\0")[:32])
        self.assertEqual(deduper.db.commits, 2)  # every 4 saved videos, like the normal loop
        self.assertEqual(deduper.db.threads, {threading.main_thread().name})  # DB only touched by the main thread

    def test_gpu_fallback_counts_as_cpu(self):
        hashes = ["gpu0", "gpu1", "cpu0"]
        with mock.patch.object(dedup, "last_decode_used_gpu", lambda: False):  # e.g. NVDEC couldn't decode them
            stats, decoded_on = self._run(self._deduper(gpu_available=True), hashes)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (0, 3))
        self.assertTrue(decoded_on["gpu0"] and decoded_on["gpu1"])  # they were still sent to the GPU worker

    def test_no_cuda_device_all_cpu(self):
        hashes = ["gpu0", "cpu0", "gpu1"]
        stats, decoded_on = self._run(self._deduper(gpu_available=False), hashes)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (0, 3))
        self.assertFalse(any(decoded_on.values()))

    def test_share_work_idle_gpu_helps(self):
        hashes = [f"cpu{i}" for i in range(8)]
        stats, decoded_on = self._run(self._deduper(gpu_available=True), hashes, share_work=True, delay=0.05)
        self.assertEqual(stats.success_hash_count, 8)
        self.assertGreater(stats.gpu_decoded_count, 0)  # the idle GPU worker helped with CPU videos
        self.assertGreater(stats.cpu_decoded_count, 0)

    def test_share_work_cpu_never_takes_gpu_videos(self):
        hashes = [f"gpu{i}" for i in range(8)]
        stats, decoded_on = self._run(self._deduper(gpu_available=True), hashes, share_work=True, delay=0.05)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (8, 0))
        self.assertTrue(all(decoded_on.values()))

    def test_failures(self):
        hashes = ["cpu0", "apierr0", "gpu0", "cpu1"]
        stats, _ = self._run(self._deduper(gpu_available=True), hashes, fail=("cpu1",))
        self.assertEqual(stats.success_hash_count, 2)
        self.assertEqual(stats.failed_from_api_errors_count, 1)
        self.assertEqual(stats.failed_from_phash_count, 1)
        self.assertEqual((stats.gpu_decoded_count, stats.cpu_decoded_count), (1, 2))  # API error: never decoded

    def test_skip_returns_early(self):
        deduper = self._deduper(gpu_available=True)
        deduper.should_skip_step_fn = lambda: True
        stats, _ = self._run(deduper, ["cpu0", "gpu0"])
        self.assertEqual(stats.success_hash_count, 0)

    def test_summary_message(self):
        self.assertEqual(
            dedup.gpu_decode_auto_summary(1, 0), "GPU decode (auto): 1 video decoded on the GPU, 0 on the CPU."
        )
        self.assertEqual(
            dedup.gpu_decode_auto_summary(4, 13), "GPU decode (auto): 4 videos decoded on the GPU, 13 on the CPU."
        )


if __name__ == "__main__":
    unittest.main()
