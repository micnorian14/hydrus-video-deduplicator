from __future__ import annotations

import ctypes
import io
import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import av
from hvdaccelerators import vpdq

try:
    from av.codec.hwaccel import HWAccel
except ImportError:  # PyAV without hardware decoding support: GPU decode is reported as unavailable
    HWAccel = None

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Annotated, TypeAlias

    from .typing_utils import ValueRange

log = logging.getLogger(__name__)
# Keep frame-level vpdqpy messages off the default console (historical).
log.setLevel(logging.CRITICAL)
# GPU fallback is a user-facing warning and must follow the app logging config
# (--verbose on the CLI; the GUI's default INFO logging).
_gpu_log = logging.getLogger("hydrusvideodeduplicator.vpdqpy")

# The dimensions of the image after downscaling for pdq
DOWNSCALE_DIMENSIONS = 512

VpdqHash: TypeAlias = vpdq.VpdqHash

# Per thread: whether the last computeHash() really decoded on the GPU (see Vpdq.last_decode_used_gpu).
_decode_state = threading.local()


class NotDecodedOnGpuError(Exception):
    """GPU decoding was asked for, but FFmpeg fell back to (single-threaded) software decoding for this video."""


def cuda_device() -> str | None:
    """GPU_DECODE_DEVICE: which NVIDIA GPU decodes (0 = the first). Unset or not a number: the default GPU."""
    value = os.environ.get("GPU_DECODE_DEVICE", "").strip()
    return value if value.isdigit() else None


# The highest frame rate trusted from a fallback guess (see frames_per_hash). Real videos stay well below this.
MAX_GUESSED_FPS = 240


def _rate_value(rate) -> float:
    """A stream rate as a float, or NaN if it's missing or unreadable (e.g. 0/0 in a WebM without a frame rate)."""
    try:
        return float(rate)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return math.nan


def _usable_guess(value: float) -> bool:
    return math.isfinite(value) and 1 <= value <= MAX_GUESSED_FPS


def _header_duration(container, stream) -> float | None:
    """Duration in seconds from the container or stream header (None if the header doesn't say). Nothing is decoded."""
    duration = None
    if container.duration:
        duration = container.duration / av.time_base
    elif stream.duration and stream.time_base:
        duration = float(stream.duration * stream.time_base)
    return duration if duration and duration > 0 else None


def measure_frame_rate(video_bytes: bytes) -> float:
    """
    Average frame rate of the first video stream, measured from its packet timestamps without decoding.
    NaN if it can't be measured. Used only for videos whose header has no usable average_rate.
    """
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            count, first, end, has_durations = 0, None, None, False
            for packet in container.demux(stream):
                if not packet.size or packet.pts is None:
                    continue
                count += 1
                stop = packet.pts + (packet.duration or 0)
                has_durations = has_durations or bool(packet.duration)
                first = packet.pts if first is None else min(first, packet.pts)
                end = stop if end is None else max(end, stop)
            if count < 2 or not stream.time_base:
                return math.nan
            span = float((end - first) * stream.time_base)
            if span <= 0:
                return math.nan
            return (count if has_durations else count - 1) / span
    except Exception:
        return math.nan


def frames_per_hash(stream, duration: float | None = None, container=None, measured_rate=None) -> int:
    """
    Hash one of every N decoded frames, about one per second of video. Returns N.

    A usable average_rate gives exactly the same N as before: round(rate) at 1 fps and above, and 1 (every frame)
    below 1 fps, e.g. a slow GIF. Only when average_rate is missing or invalid (0, NaN, None, infinite), N comes
    from the first usable estimate of: measured_rate (a number, or a function that measures it, see
    measure_frame_rate), the header's frame count / duration, PyAV's guessed_rate, base_rate. N is 1 only if none
    is usable. The duration is read from container if it isn't given.
    Before this fix such files hashed every frame (old PyAV read the rate as 0) or failed to hash (newer PyAV).
    """
    if not stream.time_base:
        return 1  # unchanged
    average = _rate_value(stream.average_rate)
    if math.isfinite(average) and average != 0:
        return round(average) if average >= 1 else 1  # unchanged
    # Header durations (GIFs) and guessed rates (variable frame rate WebMs, often 100) can be far off,
    # so the rate measured from the timestamps comes first.
    if duration is None and container is not None:
        duration = _header_duration(container, stream)
    estimates = [measured_rate, stream.frames / duration if stream.frames and duration else None]
    estimates += [getattr(stream, name, None) for name in ("guessed_rate", "base_rate")]
    for estimate in estimates:
        value = _rate_value(estimate() if callable(estimate) else estimate)
        if _usable_guess(value):
            return round(value)
        if 0 < value < 1:
            return 1  # a real slow video: every frame
    return 1


# NVIDIA's video decoder library (NVDEC). CUDA can work without it, e.g. in Docker without the "video" capability.
NVDEC_LIBRARIES = {"win32": ("nvcuvid.dll",), "linux": ("libnvcuvid.so.1", "libnvcuvid.so")}


class Vpdq:
    @staticmethod
    def get_video_bytes(video_file: Path | str | bytes) -> bytes:
        """Get the bytes of a video"""
        video = bytes()
        if isinstance(video_file, (Path, str)):
            if not Path(video_file).is_file():
                raise ValueError("Failed to get video file bytes. Video does not exist")

            try:
                with open(str(video_file), "rb") as file:
                    video = file.read()
            except OSError as exc:
                raise ValueError("Failed to get video file bytes. Invalid object type.") from exc
        elif isinstance(video_file, bytes):
            video = video_file
        else:
            raise ValueError("Failed to get video file bytes. Invalid object type.")

        return video

    @staticmethod
    def match_hash(
        query_features: VpdqHash,
        target_features: VpdqHash,
        distance_tolerance: float = 31.0,
    ):
        """Get the similarity of two videos by comparing their list of features"""
        return vpdq.matchHash(query_features, target_features, int(distance_tolerance))

    @staticmethod
    def cuda_decode_available() -> bool:
        """True if a CUDA device for GPU decoding can be opened with this PyAV (17 or newer)."""
        if HWAccel is None:
            return False
        try:
            HWAccel(device_type="cuda", device=cuda_device(), allow_software_fallback=False, is_hw_owned=True).create(
                av.Codec("h264", "r")
            )
        except Exception:
            return False
        return True

    @staticmethod
    def nvdec_library_available() -> bool:
        """True if NVIDIA's video decoder library can be loaded. Only loads it; nothing is decoded.

        Always True on macOS and other systems, where the CUDA check already fails.
        """
        platform = "linux" if sys.platform.startswith("linux") else sys.platform
        names = NVDEC_LIBRARIES.get(platform)
        if names is None:
            return True
        for name in names:
            try:
                ctypes.CDLL(name)
            except OSError:
                continue
            return True
        return False

    @staticmethod
    def probe_video_stream(video_bytes: bytes) -> tuple[str, int, int] | None:
        """Codec name, width and height of the first video stream, read from the container header only.

        Nothing is decoded (a few milliseconds even for large files). Returns None if it can't be read.
        """
        try:
            with av.open(io.BytesIO(video_bytes)) as container:
                if not container.streams.video:
                    return None
                codec_context = container.streams.video[0].codec_context
                return (codec_context.name or "", int(codec_context.width or 0), int(codec_context.height or 0))
        except Exception:
            return None

    @staticmethod
    def probe_video_details(video_bytes: bytes) -> tuple[str, int, int, float | None] | None:
        """probe_video_stream() plus the duration in seconds (None if the header doesn't say). Nothing is decoded."""
        try:
            with av.open(io.BytesIO(video_bytes)) as container:
                if not container.streams.video:
                    return None
                stream = container.streams.video[0]
                codec_context = stream.codec_context
                return (
                    codec_context.name or "",
                    int(codec_context.width or 0),
                    int(codec_context.height or 0),
                    _header_duration(container, stream),
                )
        except Exception:
            return None

    @staticmethod
    def frame_extract_pyav(video_bytes: bytes, gpu_decode: bool = False) -> Iterator[av.VideoFrame]:
        """Extract frames from video

        If gpu_decode is True, decoding uses the NVIDIA hardware decoder (CUDA/NVDEC). Hashes are
        identical for the codecs tested (H.264, HEVC, VP8, VP9). MPEG-4 and WMV3 can differ by a few
        bits but still score as a full match, so duplicate detection isn't affected.
        """
        open_kwargs = {}
        if gpu_decode:
            if HWAccel is None:
                raise RuntimeError("This PyAV has no hardware decoding support.")
            # is_hw_owned keeps decoded frames in GPU memory, so only the frames we keep (one per second)
            # are copied back to the CPU by reformat() below, instead of every decoded frame.
            open_kwargs["hwaccel"] = HWAccel(
                device_type="cuda", device=cuda_device(), allow_software_fallback=True, is_hw_owned=True
            )
        with av.open(io.BytesIO(video_bytes), **open_kwargs) as container:
            # Check for video in video container
            video_streams = container.streams.video
            if video_streams is None or len(video_streams) < 1:
                log.error("Video stream not found.")
                raise ValueError("Video stream not found.")

            video = container.streams.video[0]
            # is_hwaccel is False when this FFmpeg has no CUDA decoder for the codec. Then it's software decoding,
            # so make it multi-threaded like CPU decoding (the GPU path would otherwise decode on one core).
            on_gpu = gpu_decode and video.codec_context.is_hwaccel
            if not on_gpu:
                video.thread_type = "AUTO"
            _decode_state.used_gpu = on_gpu
            checked_first_frame = not on_gpu

            # Same for CPU and GPU decoding. Slow videos (below 1 fps, like some GIFs) hash every frame.
            average_fps = frames_per_hash(
                video, container=container, measured_rate=lambda: measure_frame_rate(video_bytes)
            )
            if average_fps == 1:
                log.warning("Average FPS is missing or less than 1. Every frame will be hashed.")

            # The following is a very overly complex "for loop" in order to decode frames from the video.
            # Why not use a for loop? Because for some reason av>=12 will sometimes throw an
            # av.error.InvalidDataError error in container.decode(). I'm not sure why this happens.
            # Normally this loop would be written as `for frame_index, frame in container.decode(video)`,
            # but to catch the exception I need to wrap the decode call with try/catch and iterate using next().
            frame_generator = container.decode(video)
            frame_index = 0
            while True:
                try:
                    frame = next(frame_generator)
                    if not checked_first_frame:
                        checked_first_frame = True
                        if frame.format.name != "cuda":
                            # NVDEC can't decode this stream (e.g. H.264 4:4:4, or a profile or size the card
                            # doesn't support), so FFmpeg fell back to single-threaded software decoding.
                            # Stop before anything is hashed; computeHash() redoes it with multi-threaded CPU decoding.
                            raise NotDecodedOnGpuError(f"no hardware decoding for this video ({frame.format.name})")
                    if frame_index % average_fps == 0:
                        yield frame.reformat(
                            width=DOWNSCALE_DIMENSIONS,
                            height=DOWNSCALE_DIMENSIONS,
                            format="rgb24",
                            interpolation=av.video.reformatter.Interpolation.POINT,
                        )
                    frame_index += 1
                except StopIteration:
                    break
                except av.error.InvalidDataError as exc:
                    log.error(f"Skipping bad frame at index {frame_index}: {exc}")
                    frame_index += 1

    @staticmethod
    def computeHash(video_file: Path | str | bytes, num_threads: int = 0, gpu_decode: bool = False) -> VpdqHash:
        """Perceptually hash video from a file path or the bytes"""
        video = Vpdq.get_video_bytes(video_file)
        if video is None:
            raise ValueError

        _decode_state.used_gpu = None
        if gpu_decode:
            try:
                return Vpdq._hash_video_bytes(video, num_threads, gpu_decode=True)
            except Exception as exc:
                # Anything the GPU can't handle is retried with normal CPU decoding.
                _gpu_log.warning(f"GPU decode failed, falling back to CPU decode: {exc}")
        return Vpdq._hash_video_bytes(video, num_threads, gpu_decode=False)

    @staticmethod
    def last_decode_used_gpu() -> bool | None:
        """True if the last computeHash() on this thread decoded on the GPU, False if on the CPU (including a GPU
        request that fell back to the CPU), None if unknown (e.g. the video failed before decoding started)."""
        return getattr(_decode_state, "used_gpu", None)

    @staticmethod
    def _hash_video_bytes(video: bytes, num_threads: int, gpu_decode: bool) -> VpdqHash:
        # Average FPS is used by vpdq to calculate the timestamp, but we completely discard
        # the timestamp so this value doesn't matter.
        average_fps = 1
        hasher = vpdq.VideoHasher(average_fps, DOWNSCALE_DIMENSIONS, DOWNSCALE_DIMENSIONS, num_threads)
        for frame in Vpdq.frame_extract_pyav(video, gpu_decode=gpu_decode):
            # Note: hash_frame will block if vpdq's internal frame queue is full. This is necessary,
            # otherwise if hashing gets too far behind decoding there will be an insane amount of memory
            # used to hold the raw frames.
            hasher.hash_frame(bytes(frame.planes[0]))
        return hasher.finish()

    @staticmethod
    def is_similar(
        vpdq_features1: VpdqHash,
        vpdq_features2: VpdqHash,
        threshold: Annotated[float, ValueRange(0.0, 100.0)] = 75.0,
    ) -> tuple[bool, float]:
        """Check if video is similar by comparing their list of features
        Threshold is minimum similarity to be considered similar
        """
        similarity = Vpdq.match_hash(query_features=vpdq_features1, target_features=vpdq_features2)
        return similarity >= threshold, similarity
