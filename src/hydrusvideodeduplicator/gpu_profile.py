"""
--gpu-decode auto: learn how fast this computer's CPU and GPU hash each kind of video, and remember it.

The profile is a small JSON file next to the dedupe database (gpu_profile.json), never inside it. Deleting it, or
--reset-gpu-profile, only makes auto mode start again from its fixed rules (prefers_gpu_decode). Hashes are identical
on the GPU and the CPU, so the profile only changes speed.

Each hashed video is timed on the worker that hashed it and grouped by codec and resolution (a "kind"). Time is
measured per second of video, so short and long videos of the same kind compare fairly. A kind's speed on a device is
only used after MIN_SAMPLES videos were timed there; until then the fixed rules decide.
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from .hashing import prefers_gpu_decode

PROFILE_FILE_NAME = "gpu_profile.json"
PROFILE_VERSION = 1
# A device's speed for a kind of video is only used after this many videos of that kind were timed on it.
MIN_SAMPLES = 5
# Retuning: the median of the last RECENT_SAMPLES timings (once there are at least MIN_RECENT) moves the estimate
# EMA_WEIGHT of the way there, and never by more than a factor of MAX_STEP. So a few odd videos can't swing it.
RECENT_SAMPLES = 9
MIN_RECENT = 3
EMA_WEIGHT = 0.2
MAX_STEP = 2.0
# Videos shorter than this are timed as if they were this long (setup time dominates very short clips).
MIN_DURATION_S = 0.5
# Load-aware sharing: an idle worker takes a video meant for the other (busy) worker only if its own estimate for that
# kind is at most this many times the other's. The CPU limit is tight: a heavy GPU video (e.g. H.264 4K) on the CPU
# was measured to slow the whole run down.
GPU_HELP_MAX_RATIO = 2.0
CPU_HELP_MAX_RATIO = 1.2
# Exploring: an idle CPU worker may time a GPU video of a kind it hasn't measured yet, but only while at least this
# many videos are waiting for the GPU worker and more are still being downloaded.
CPU_EXPLORE_MIN_BACKLOG = 2

# Resolution groups by pixel count (portrait and landscape count the same). A video belongs to the first group whose
# pixel count it doesn't exceed by more than 10%.
RESOLUTION_BUCKETS = (
    (640 * 360, "360p"),
    (1280 * 720, "720p"),
    (1920 * 1080, "1080p"),
    (2560 * 1440, "1440p"),
    (3840 * 2160, "2160p"),
)

# GpuProfile.load() results
LOADED = "loaded"
NEW = "new"
RESET = "reset"
CORRUPT = "corrupt"
HARDWARE_CHANGED = "hardware_changed"

CPU = "cpu"
GPU = "gpu"


def resolution_bucket(width: int | None, height: int | None) -> str:
    pixels = max(width or 0, 0) * max(height or 0, 0)
    if pixels == 0:
        return "unknown"
    for limit, name in RESOLUTION_BUCKETS:
        if pixels <= limit * 11 // 10:
            return name
    return "larger"


def codec_group(codec: str | None) -> str:
    codec = (codec or "").strip().lower()
    if codec == "h265":
        return "hevc"
    return codec or "unknown"


def bucket_key(codec: str | None, width: int | None, height: int | None) -> str:
    return f"{codec_group(codec)}:{resolution_bucket(width, height)}"


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as file:
                for line in file:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _gpu_name_and_driver(device: str | None) -> tuple[str, str]:
    """The NVIDIA GPU's name and driver version from nvidia-smi, or empty strings if it can't be run."""
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", device or "0"],
            capture_output=True,
            text=True,
            timeout=5,
            **kwargs,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, _, driver = out.stdout.strip().splitlines()[0].partition(",")
            return name.strip(), driver.strip()
    except Exception:
        pass
    return "", ""


def hardware_fingerprint(hash_threads: int) -> dict:
    """What the speeds in a profile depend on. If any of it changes, the profile starts over."""
    from .vpdqpy.vpdqpy import cuda_device

    try:
        import av

        pyav = str(av.__version__)
    except Exception:
        pyav = ""
    device = cuda_device()
    gpu, driver = _gpu_name_and_driver(device)
    return {
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count() or 0,
        "gpu": gpu,
        "driver": driver,
        "gpu_device": device or "",
        "pyav": pyav,
        "hash_threads": int(hash_threads),
    }


def remove_profile(path: Path | str | None) -> bool:
    """Delete the profile file. True if it was there and is gone now. Never raises."""
    if path is None:
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


class DeviceStats:
    """The speed of one device for one kind of video: seconds of hashing per second of video."""

    __slots__ = ("estimate", "samples", "recent")

    def __init__(self, estimate: float | None = None, samples: int = 0):
        self.estimate = estimate
        self.samples = samples
        self.recent: deque[float] = deque(maxlen=RECENT_SAMPLES)

    def ready(self) -> bool:
        return self.estimate is not None and self.samples >= MIN_SAMPLES

    def add(self, cost: float) -> None:
        self.recent.append(cost)
        self.samples += 1
        if self.estimate is None:
            if len(self.recent) >= MIN_SAMPLES:
                self.estimate = statistics.median(self.recent)
            return
        if len(self.recent) < MIN_RECENT:
            return
        target = statistics.median(self.recent)
        target = min(max(target, self.estimate / MAX_STEP), self.estimate * MAX_STEP)
        self.estimate += EMA_WEIGHT * (target - self.estimate)


def _valid_number(value, positive_int: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    if positive_int:
        return isinstance(value, int) and value >= MIN_SAMPLES
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _parse_buckets(data) -> dict[str, dict[str, DeviceStats]]:
    """The speeds in a loaded profile. Raises ValueError if the file isn't a valid profile."""
    if not isinstance(data, dict) or data.get("version") != PROFILE_VERSION:
        raise ValueError("not a version 1 profile")
    buckets = data.get("buckets")
    if not isinstance(data.get("fingerprint"), dict) or not isinstance(buckets, dict):
        raise ValueError("missing fingerprint or buckets")
    out: dict[str, dict[str, DeviceStats]] = {}
    for key, devices in buckets.items():
        if not isinstance(devices, dict):
            raise ValueError(f"bad entry {key!r}")
        entry = {}
        for device in (CPU, GPU):
            stats = devices.get(device)
            if stats is None:
                continue
            if not isinstance(stats, dict):
                raise ValueError(f"bad entry {key!r}")
            estimate, samples = stats.get("seconds_per_video_second"), stats.get("samples")
            if not _valid_number(estimate) or not _valid_number(samples, positive_int=True):
                raise ValueError(f"bad entry {key!r}")
            entry[device] = DeviceStats(float(estimate), samples)
        if entry:
            out[key] = entry
    return out


class GpuProfile:
    """Thread-safe: the hashing workers record timings while the download thread asks where to send videos."""

    def __init__(self, path: Path | str | None, fingerprint: dict):
        self.path = Path(path) if path is not None else None
        self.fingerprint = dict(fingerprint)
        self._stats: dict[str, dict[str, DeviceStats]] = {}
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._dirty = False
        self.save_error: str | None = None
        self.save_error_reported = False

    @classmethod
    def load(cls, path: Path | str | None, fingerprint: dict, reset: bool = False) -> tuple[GpuProfile, str]:
        """The saved profile and LOADED, or an empty one and why (NEW, RESET, CORRUPT, HARDWARE_CHANGED).

        Never raises: anything wrong with the file just means starting from the fixed rules. A bad file is
        overwritten at the next save.
        """
        profile = cls(path, fingerprint)
        if reset:
            remove_profile(path)
            return profile, RESET
        if profile.path is None:
            return profile, NEW
        try:
            with open(profile.path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return profile, NEW
        except Exception:
            return profile, CORRUPT
        try:
            stats = _parse_buckets(data)
        except Exception:
            return profile, CORRUPT
        if data["fingerprint"] != profile.fingerprint:
            return profile, HARDWARE_CHANGED
        profile._stats = stats
        return profile, LOADED

    def record(
        self, codec: str | None, width: int | None, height: int | None, duration: float | None, on_gpu: bool, seconds
    ) -> bool:
        """Add one video's hashing time on the GPU or CPU worker. False (ignored) if it can't be used."""
        if duration is None or seconds is None:
            return False
        try:
            duration, seconds = float(duration), float(seconds)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(duration) and math.isfinite(seconds)) or duration <= 0 or seconds <= 0:
            return False
        cost = seconds / max(duration, MIN_DURATION_S)
        key = bucket_key(codec, width, height)
        with self._lock:
            self._stats.setdefault(key, {}).setdefault(GPU if on_gpu else CPU, DeviceStats()).add(cost)
            self._dirty = True
        return True

    def estimates(self, codec: str | None, width: int | None, height: int | None) -> tuple[float | None, float | None]:
        """(CPU, GPU) seconds per second of video for this kind, each None until it has MIN_SAMPLES timings."""
        with self._lock:
            entry = self._stats.get(bucket_key(codec, width, height), {})
            return tuple(
                entry[device].estimate if device in entry and entry[device].ready() else None for device in (CPU, GPU)
            )

    def samples(self, codec: str | None, width: int | None, height: int | None, on_gpu: bool) -> int:
        with self._lock:
            stats = self._stats.get(bucket_key(codec, width, height), {}).get(GPU if on_gpu else CPU)
            return stats.samples if stats else 0

    def prefers_gpu(self, codec: str | None, width: int | None, height: int | None) -> bool:
        """Where to send a video: the faster device once both are measured for its kind, else the fixed rules."""
        cpu, gpu = self.estimates(codec, width, height)
        if cpu is not None and gpu is not None:
            return gpu < cpu
        return prefers_gpu_decode(codec, width, height)

    def gpu_may_help(self, info) -> bool:
        """An idle GPU worker may take this CPU video unless it's measured to be much slower on the GPU."""
        if info is None:
            return True
        cpu, gpu = self.estimates(*info[:3])
        if cpu is None or gpu is None:
            return True
        return gpu <= cpu * GPU_HELP_MAX_RATIO

    def cpu_may_help(self, info, explore: bool) -> bool:
        """An idle CPU worker may take this GPU video if it's measured to be about as fast on the CPU, or (explore)
        to time a kind it hasn't measured yet."""
        if info is None:
            return False
        cpu, gpu = self.estimates(*info[:3])
        if cpu is not None and gpu is not None:
            return cpu <= gpu * CPU_HELP_MAX_RATIO
        return explore and cpu is None and self.samples(*info[:3], on_gpu=False) < MIN_SAMPLES

    def learned_kinds(self) -> int:
        """How many kinds of video have speeds measured on both devices."""
        with self._lock:
            return sum(1 for entry in self._stats.values() if all(d in entry and entry[d].ready() for d in (CPU, GPU)))

    def to_json(self) -> dict:
        with self._lock:
            buckets = {}
            for key in sorted(self._stats):
                entry = {
                    device: {"seconds_per_video_second": round(stats.estimate, 6), "samples": stats.samples}
                    for device, stats in sorted(self._stats[key].items())
                    if stats.ready()
                }
                if entry:
                    buckets[key] = entry
            return {"version": PROFILE_VERSION, "fingerprint": dict(self.fingerprint), "buckets": buckets}

    def save(self) -> bool:
        """Write the profile if anything new was learned (a temporary file, then replaced in one step, so a crash
        can't leave half a file). Never raises: False and save_error if it couldn't be written."""
        with self._save_lock:
            with self._lock:
                if not self._dirty or self.path is None:
                    return True
                self._dirty = False
            data = self.to_json()
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=1)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(tmp, self.path)
            except Exception as exc:
                with self._lock:
                    self._dirty = True
                remove_profile(tmp)
                self.save_error = str(exc) or type(exc).__name__
                return False
            self.save_error = None
            return True
