from __future__ import annotations

import ctypes
import io
import logging
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
log.setLevel(logging.CRITICAL)

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
                duration = None
                if container.duration:
                    duration = container.duration / av.time_base
                elif stream.duration and stream.time_base:
                    duration = float(stream.duration * stream.time_base)
                return (
                    codec_context.name or "",
                    int(codec_context.width or 0),
                    int(codec_context.height or 0),
                    duration if duration and duration > 0 else None,
                )
        except Exception:
            return None

    @staticmethod
    def frame_extract_pyav(video_bytes: bytes, gpu_decode: bool = False) -> Iterator[av.VideoFrame]:
        """Extract frames from video

        If gpu_decode is True, decoding uses the NVIDIA hardware decoder (CUDA/NVDEC). Decoded frames are
        identical to CPU decoding, so the resulting perceptual hashes are identical too.
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

            average_fps: int = 1
            # Some videos, like small GIFs, will have a NoneType FPS
            if not video.time_base or float(video.average_rate) < 1:
                log.warning("Average FPS is None or less than 1. Every frame will be hashed.")
            else:
                average_fps = round(float(video.average_rate))

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
                log.warning(f"GPU decode failed, falling back to CPU decode: {exc}")
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
