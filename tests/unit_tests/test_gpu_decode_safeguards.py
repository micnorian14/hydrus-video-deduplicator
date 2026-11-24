from __future__ import annotations

import threading
import unittest
from unittest import mock

from rich.text import Text

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.config import Config, GpuDecodeArgError, normalize_gpu_decode_args
from hydrusvideodeduplicator.vpdqpy import vpdqpy
from hydrusvideodeduplicator.vpdqpy.vpdqpy import Vpdq

from .test_gpu_decode_auto import _FakeDb, _FakePhash

GIB = 1024**3


class TestNvdecLibraryCheck(unittest.TestCase):
    def _check(self, platform, loadable):
        tried = []

        def cdll(name):
            tried.append(name)
            if name not in loadable:
                raise OSError(name)
            return object()

        with mock.patch.object(vpdqpy.sys, "platform", platform), mock.patch.object(vpdqpy.ctypes, "CDLL", cdll):
            return Vpdq.nvdec_library_available(), tried

    def test_windows(self):
        self.assertEqual(self._check("win32", {"nvcuvid.dll"}), (True, ["nvcuvid.dll"]))
        self.assertEqual(self._check("win32", set()), (False, ["nvcuvid.dll"]))

    def test_linux(self):
        self.assertEqual(self._check("linux", {"libnvcuvid.so.1"}), (True, ["libnvcuvid.so.1"]))
        self.assertEqual(self._check("linux", {"libnvcuvid.so"}), (True, ["libnvcuvid.so.1", "libnvcuvid.so"]))
        self.assertEqual(self._check("linux", set()), (False, ["libnvcuvid.so.1", "libnvcuvid.so"]))

    def test_macos_not_checked(self):
        self.assertEqual(self._check("darwin", set()), (True, []))


class TestGpuDecodeStatus(unittest.TestCase):
    def _status(self, cuda, nvdec, auto=False):
        lines = []  # what reaches the console, as plain text

        def fake_print(*args, **kwargs):
            lines.append(Text.from_markup(str(args[0])).plain if args else "")

        with (
            mock.patch.object(Vpdq, "cuda_decode_available", staticmethod(lambda: cuda)),
            mock.patch.object(Vpdq, "nvdec_library_available", staticmethod(lambda: nvdec)),
            mock.patch.object(dedup, "console_print", lambda text, **kw: lines.append(text.plain)),
            mock.patch.object(dedup, "print", fake_print),
        ):
            status = dedup.print_gpu_decode_status(auto=auto)
        return status, "".join(line + "\n" for line in lines)

    def test_missing_nvdec(self):
        status, out = self._status(cuda=True, nvdec=False)
        self.assertEqual(status, dedup.GPU_STATUS_NO_NVDEC)
        self.assertEqual(
            out,
            "WARNING: GPU decode is off: your NVIDIA driver is missing video decoding support.\n"
            "Decoding on the CPU instead. Fix: update your NVIDIA driver, or see the README.\n\n",
        )
        self.assertEqual(self._status(cuda=True, nvdec=False, auto=True)[0], dedup.GPU_STATUS_NO_NVDEC)

    def test_on_and_no_device_unchanged(self):
        self.assertEqual(self._status(cuda=True, nvdec=True), (dedup.GPU_STATUS_ON, "Using GPU decode (all).\n\n"))
        self.assertEqual(
            self._status(cuda=True, nvdec=True, auto=True), (dedup.GPU_STATUS_ON, "Using GPU decode (auto).\n\n")
        )
        status, out = self._status(cuda=False, nvdec=False)
        self.assertEqual(status, dedup.GPU_STATUS_NO_DEVICE)
        self.assertEqual(out, "WARNING: --gpu-decode all: no CUDA device found, decoding on the CPU.\n\n")

    def test_plain_flag_turned_off_without_nvdec(self):
        for status, auto, expected in [
            (dedup.GPU_STATUS_NO_NVDEC, False, False),
            (dedup.GPU_STATUS_NO_DEVICE, False, True),  # unchanged: keeps trying per video
            (dedup.GPU_STATUS_NO_NVDEC, True, "auto"),
        ]:
            with (
                mock.patch.object(dedup, "print_gpu_decode_status", lambda auto=False, s=status: s),
                mock.patch.object(dedup, "HydrusPageLogger"),
                mock.patch.object(dedup.HydrusVideoDeduplicator, "get_search_tags", lambda self, q: []),
            ):
                deduper = dedup.HydrusVideoDeduplicator(None, None, gpu_decode="auto" if auto else True)
            self.assertEqual(deduper.gpu_decode, expected)
            self.assertFalse(deduper.gpu_decode_available)


class TestCudaDevice(unittest.TestCase):
    def test_env(self):
        for value, expected in [(None, None), ("", None), ("0", "0"), (" 1 ", "1"), ("gpu1", None), ("-1", None)]:
            env = {} if value is None else {"GPU_DECODE_DEVICE": value}
            with mock.patch.dict(vpdqpy.os.environ, env, clear=False):
                if value is None:
                    vpdqpy.os.environ.pop("GPU_DECODE_DEVICE", None)
                self.assertEqual(vpdqpy.cuda_device(), expected, value)


class TestArgValues(unittest.TestCase):
    def test_on_off_values(self):
        # Older spellings still work, undocumented: on/true/1/yes mean all, false/0/no mean off.
        for value in ["all", "on", "TRUE", "1", "yes"]:
            self.assertEqual(normalize_gpu_decode_args(["--gpu-decode", value]), ["--gpu-decode", "all"])
            self.assertEqual(normalize_gpu_decode_args([f"--gpu-decode={value}"]), ["--gpu-decode", "all"])
        for value in ["off", "False", "0", "no"]:
            self.assertEqual(normalize_gpu_decode_args(["--gpu-decode", value]), ["--gpu-decode", "off"])
            self.assertEqual(normalize_gpu_decode_args([f"--gpu-decode={value}"]), ["--gpu-decode", "off"])
        # No value, or an unknown one: a clear error instead of a guess
        for args in (["--gpu-decode", "--verbose"], ["--gpu-decode=maybe"]):
            with self.assertRaises(GpuDecodeArgError, msg=args):
                normalize_gpu_decode_args(args)

    def test_unknown_env_value(self):
        for value in ["enabled", "2", "Maybe"]:
            config = Config._load({"GPU_DECODE": value})
            self.assertEqual(
                (config.gpu_decode, config.gpu_decode_auto, config.gpu_decode_invalid), (False, False, value)
            )
        for value in ["all", "1", "true", "YES", "on", "0", "false", "no", "off", "", "Auto"]:
            self.assertIsNone(Config._load({"GPU_DECODE": value}).gpu_decode_invalid, value)
        self.assertIsNone(Config._load({}).gpu_decode_invalid)


class TestRamSafeguard(unittest.TestCase):
    def test_ready_limit(self):
        mib = 1024**2
        self.assertEqual(dedup.auto_ready_limit(None, 10 * GIB), dedup.AUTO_READY_FILES)  # unknown: unchanged
        # small videos count as 512 MB: need = 2 GB + 1 GB = 3 GB
        self.assertEqual(dedup.auto_ready_limit(5 * GIB, 100 * mib), 2)
        self.assertEqual(dedup.auto_ready_limit(5 * GIB - 1, 100 * mib), 1)
        self.assertEqual(dedup.auto_ready_limit(3 * GIB, 100 * mib), 1)
        self.assertEqual(dedup.auto_ready_limit(3 * GIB - 1, 100 * mib), 0)
        # 3 GB videos: need = 2 GB + 6 GB = 8 GB
        self.assertEqual(dedup.auto_ready_limit(10 * GIB, 3 * GIB), 2)
        self.assertEqual(dedup.auto_ready_limit(9 * GIB, 3 * GIB), 1)
        self.assertEqual(dedup.auto_ready_limit(7 * GIB, 3 * GIB), 0)

    def test_hash_alone(self):
        self.assertTrue(dedup.hash_alone(3 * GIB, 8 * GIB))
        self.assertFalse(dedup.hash_alone(2 * GIB, 8 * GIB))
        self.assertFalse(dedup.hash_alone(100 * GIB, None))


class TestWorkerSettings(unittest.TestCase):
    def test_threads_and_sharing(self):
        # (CPU worker threads, GPU worker threads, GPU worker helps with CPU videos)
        self.assertEqual(dedup.auto_worker_settings(1, -2), (1, 1, False))
        self.assertEqual(dedup.auto_worker_settings(2, -2), (1, 1, False))
        self.assertEqual(dedup.auto_worker_settings(3, -2), (1, 1, True))
        self.assertEqual(dedup.auto_worker_settings(4, -2), (1, 1, True))
        self.assertEqual(dedup.auto_worker_settings(12, 6), (5, 1, True))
        self.assertEqual(dedup.auto_worker_settings(12, 0), (0, 0, True))  # 0 = all threads: not capped, as before
        with mock.patch.object(dedup, "AUTO_SHARE_WORK", False):
            self.assertEqual(dedup.auto_worker_settings(12, -2), (8, 2, False))

    def test_split_never_exceeds_the_job_count(self):
        for cpus in range(1, 65):
            for job_count in [*range(-cpus - 1, 0), *range(1, 2 * cpus + 2)]:
                cpu_threads, gpu_threads, _ = dedup.auto_worker_settings(cpus, job_count)
                budget = dedup.resolve_job_count(job_count, cpus)
                self.assertGreaterEqual(cpu_threads, 1)
                self.assertGreaterEqual(gpu_threads, 1)
                # A budget of 1 can't be split; otherwise the two workers together stay within it.
                self.assertLessEqual(cpu_threads + gpu_threads, max(budget, 2), (cpus, job_count))
                if budget >= 2:
                    self.assertEqual(cpu_threads + gpu_threads, budget, (cpus, job_count))

    def test_logical_cpu_count(self):
        self.assertGreaterEqual(dedup.logical_cpu_count(), 1)


class TestAutoPipelineSafeguards(unittest.TestCase):
    def _deduper(self, gpu_available=True, skip=None):
        deduper = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        deduper.db = _FakeDb()
        deduper.client = None
        deduper.job_count = 1
        deduper.gpu_decode = "auto"
        deduper.gpu_decode_auto = True
        deduper.gpu_decode_available = gpu_available
        deduper.page_logger = None
        deduper.update_progress_callback = None
        deduper.should_skip_step_fn = skip
        return deduper

    def _run(self, deduper, hashes, memory, delay=0.02, cpus=12):
        lock = threading.Lock()
        counts = {"fetched": 0, "computed": 0, "active": 0, "max_active": 0, "max_in_memory_at_fetch": 0, "threads": []}

        def fetch(_self, file_hash):
            with lock:
                in_memory = counts["fetched"] - counts["computed"]
                counts["max_in_memory_at_fetch"] = max(counts["max_in_memory_at_fetch"], in_memory)
                counts["fetched"] += 1
            return file_hash.encode()

        def compute(video, num_threads, gpu_decode):
            with lock:
                counts["active"] += 1
                counts["max_active"] = max(counts["max_active"], counts["active"])
                counts["threads"].append((gpu_decode, num_threads))
            threading.Event().wait(delay)
            with lock:
                counts["active"] -= 1
                counts["computed"] += 1
            return _FakePhash(video.ljust(32, b"\0")[:32])

        with (
            mock.patch.object(dedup.FileHasher, "_fetch_file", fetch),
            mock.patch.object(dedup, "compute_phash", compute),
            mock.patch.object(dedup, "video_prefers_gpu_decode", lambda video: video.startswith(b"gpu")),
            mock.patch.object(dedup, "memory_status", lambda: memory),
            mock.patch.object(dedup, "logical_cpu_count", lambda: cpus),
        ):
            stats = deduper.add_perceptual_hashes_to_db(hashes)
        return stats, counts

    HASHES = [f"{kind}{i}" for i in range(4) for kind in ("gpu", "cpu")]

    def test_low_ram_one_video_at_a_time(self):
        stats, counts = self._run(self._deduper(), self.HASHES, memory=(1 * GIB, 8 * GIB))
        self.assertEqual(stats.success_hash_count, len(self.HASHES))  # no deadlock
        self.assertEqual(counts["max_in_memory_at_fetch"], 0)  # each download waited until nothing was in memory

    def test_huge_videos_hashed_alone(self):
        stats, counts = self._run(self._deduper(), self.HASHES, memory=(100 * GIB, 8))  # every video > 25% of RAM
        self.assertEqual(stats.success_hash_count, len(self.HASHES))
        self.assertEqual(counts["max_active"], 1)

    def test_plenty_of_ram_both_workers(self):
        stats, counts = self._run(self._deduper(), self.HASHES, memory=(100 * GIB, 128 * GIB), delay=0.05)
        self.assertEqual(stats.success_hash_count, len(self.HASHES))
        self.assertEqual(counts["max_active"], 2)

    def test_unknown_ram(self):
        stats, _ = self._run(self._deduper(), self.HASHES, memory=None)
        self.assertEqual(stats.success_hash_count, len(self.HASHES))

    def test_weak_cpu_gpu_worker_one_thread(self):
        stats, counts = self._run(self._deduper(), self.HASHES, memory=None, cpus=2)
        self.assertEqual(stats.success_hash_count, len(self.HASHES))
        self.assertEqual({t for g, t in counts["threads"] if g}, {1})
        self.assertEqual({t for g, t in counts["threads"] if not g}, {1})  # CPU worker keeps job_count (1 here)
        self.assertEqual(stats.gpu_decoded_count, 4)  # <= 2 CPUs: no sharing, only the GPU videos

    def _hvd_threads(self):
        return [t for t in threading.enumerate() if t.name.startswith("hvd-")]

    def test_threads_joined_after_skip(self):
        calls = {"n": 0}

        def skip():
            calls["n"] += 1
            return calls["n"] > 3

        self._run(self._deduper(skip=skip), self.HASHES, memory=None, delay=0.1)
        self.assertEqual(self._hvd_threads(), [])

    def test_ctrl_c_waits_at_exit(self):
        def interrupt():
            raise KeyboardInterrupt

        with mock.patch.object(dedup.atexit, "register") as register:
            with self.assertRaises(dedup.CancelledPerceptualHashException):
                self._run(self._deduper(skip=interrupt), self.HASHES, memory=None, delay=0.05)
        joins = [c for c in register.call_args_list if c.args and c.args[0] is dedup._join_threads]
        self.assertEqual(len(joins), 1)
        self.assertEqual(joins[0].args[2], dedup.AUTO_EXIT_JOIN_S)
        dedup._join_threads(joins[0].args[1], 5.0)


class TestGpuFallbackWarning(unittest.TestCase):
    def test_gpu_fallback_warning_is_logged(self):
        dummy_hash = object()

        def _hash_video_bytes(video, num_threads, gpu_decode):
            if gpu_decode:
                raise RuntimeError("nvdec failed")
            return dummy_hash

        with mock.patch.object(Vpdq, "_hash_video_bytes", staticmethod(_hash_video_bytes)):
            with self.assertLogs("hydrusvideodeduplicator.vpdqpy", level="WARNING") as cm:
                result = Vpdq.computeHash(b"fake-video-bytes", gpu_decode=True)
        self.assertIs(result, dummy_hash)
        self.assertTrue(
            any("GPU decode failed, falling back to CPU decode: nvdec failed" in message for message in cm.output)
        )


if __name__ == "__main__":
    unittest.main()
