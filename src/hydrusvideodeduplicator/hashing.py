from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Annotated

    from .typing_utils import ValueRange

from .vpdqpy.vpdqpy import Vpdq, VpdqHash


def compute_phash(video: Path | str | bytes, num_threads: int = 0, gpu_decode: bool = False) -> VpdqHash:
    """
    Calculate the perceptual hash of a video.

    Returns the perceptual hash of the video.
    """
    phash = Vpdq.computeHash(video, num_threads, gpu_decode)
    return phash


# --gpu-decode auto: H.264 at about 1440p and above goes to the GPU. Measured by pixel count so portrait videos
# (e.g. 2160x3840) count the same as landscape ones. 90% of 2560x1440 leaves room for slightly cropped videos while
# 1080p (about 2.1 million pixels) stays well below.
GPU_DECODE_MIN_H264_PIXELS = 2560 * 1440 * 9 // 10
# Tiny HEVC videos (below about 640x360) hash faster on the CPU: setting up the GPU decoder costs more than it saves.
GPU_DECODE_MIN_HEVC_PIXELS = 640 * 360 * 9 // 10


def prefers_gpu_decode(codec: str | None, width: int | None, height: int | None) -> bool:
    """
    --gpu-decode auto: True if a video with this codec and resolution is usually hashed faster with GPU decoding.

    Fixed rules from benchmarks on real files (RTX 5070 Ti, 12 and 4 CPU cores): HEVC at about 640x360 and above, and
    H.264 at about 1440p and above. Everything else (smaller HEVC, H.264 at 1080p and below, VP8, VP9, MPEG-4, AV1,
    unknown) is usually as fast or faster on the CPU. Hashes are identical for the codecs tested (H.264, HEVC, VP8,
    VP9). MPEG-4 and WMV3 can differ by a few bits but still score as a full match, so this only affects speed.
    """
    codec = (codec or "").lower()
    pixels = max(width or 0, 0) * max(height or 0, 0)
    if codec in ("hevc", "h265"):
        return pixels >= GPU_DECODE_MIN_HEVC_PIXELS
    if codec == "h264":
        return pixels >= GPU_DECODE_MIN_H264_PIXELS
    return False


def video_prefers_gpu_decode(video: bytes) -> bool:
    """prefers_gpu_decode() for a video file's bytes. Reads only the container header; False if it can't."""
    info = Vpdq.probe_video_stream(video)
    if info is None:
        return False
    return prefers_gpu_decode(*info)


def probe_video_details(video: bytes) -> tuple[str, int, int, float | None] | None:
    """(codec, width, height, duration in seconds or None) from a video file's header, or None if it can't be read."""
    return Vpdq.probe_video_details(video)


def last_decode_used_gpu() -> bool | None:
    """Whether the last compute_phash() on this thread really decoded on the GPU (None if unknown)."""
    return Vpdq.last_decode_used_gpu()


def encode_phash_to_str(phash: VpdqHash) -> str:
    """
    Encode the perceptual hash of a video into a string.

    Returns the perceptual hash encoded as a string.
    """
    encoded_phash = str(phash)
    return encoded_phash


def decode_phash_from_str(phash_str: str) -> VpdqHash:
    """
    Encode the perceptual hash of a video into a string.

    Returns the perceptual hash encoded as a string.
    """
    return VpdqHash.from_string(phash_str)


def get_phash_similarity(
    hash_a: VpdqHash,
    hash_b: VpdqHash,
) -> Annotated[float, ValueRange(0.0, 100.0)]:
    """
    Check if video is similar by comparing their list of features
    Threshold is minimum similarity to be considered similar
    """
    similarity = Vpdq.match_hash(query_features=hash_a, target_features=hash_b)
    assert similarity >= 0.0 and similarity <= 100.0
    return similarity
