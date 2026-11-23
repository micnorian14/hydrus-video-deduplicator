"""Frame rate choice in vpdqpy.frames_per_hash (WebMs without a frame rate, slow GIFs, normal videos)."""

from __future__ import annotations

import math
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import av

from hydrusvideodeduplicator.vpdqpy import vpdqpy
from hydrusvideodeduplicator.vpdqpy.vpdqpy import Vpdq, frames_per_hash


class _ZeroOverZero:
    """Like a rate of 0/0: converting it to a float fails."""

    def __float__(self):
        raise ZeroDivisionError("0/0")


class _Stream:
    def __init__(self, average, guessed=None, base=None, frames=0, time_base=Fraction(1, 1000)):
        self.average_rate = average
        self.guessed_rate = guessed
        self.base_rate = base
        self.frames = frames
        self.time_base = time_base


def _old_frames_per_hash(stream) -> int:
    """The code before the fix (it raises for NaN, None and infinite rates, i.e. those files failed to hash)."""
    if not stream.time_base or float(stream.average_rate) < 1:
        return 1
    return round(float(stream.average_rate))


class TestFramesPerHash(unittest.TestCase):
    def test_valid_average_rate_is_unchanged(self):
        for rate in (Fraction(30000, 1001), 25, 24, 60, 1, 1.4, 1.5, 2.5, 119.88, 240, 1000, 90000):
            stream = _Stream(rate, guessed=Fraction(1000, 41), base=Fraction(1000, 41), frames=500)
            self.assertEqual(frames_per_hash(stream, 10.0), _old_frames_per_hash(stream), rate)
        self.assertEqual(frames_per_hash(_Stream(Fraction(30000, 1001))), 30)
        self.assertEqual(frames_per_hash(_Stream(90000)), 90000)  # absurd but valid: kept exactly as before

    def test_slow_rates_still_hash_every_frame(self):
        # A real 0.5 or 0.75 fps GIF: every frame, as before, even if PyAV guesses something else.
        for rate in (0.5, 0.75, Fraction(3, 4), 0.01):
            stream = _Stream(rate, guessed=30, base=30, frames=100)
            self.assertEqual(frames_per_hash(stream, 3.0), 1, rate)
            self.assertEqual(_old_frames_per_hash(stream), 1)

    def test_no_time_base_is_unchanged(self):
        self.assertEqual(frames_per_hash(_Stream(30, guessed=30, time_base=None)), 1)

    def test_missing_rate_uses_guessed_rate(self):
        # The 3 WebMs from the bug report: average 0/0 (NaN), guessed and base rate 1000/41.
        for average in (math.nan, _ZeroOverZero(), None, 0, Fraction(0), math.inf):
            stream = _Stream(average, guessed=Fraction(1000, 41), base=Fraction(1000, 41))
            self.assertEqual(frames_per_hash(stream), 24, repr(average))

    def test_missing_rate_uses_base_rate(self):
        stream = _Stream(math.nan, guessed=math.nan, base=Fraction(1000, 33))
        self.assertEqual(frames_per_hash(stream), 30)

    def test_absurd_guesses_are_skipped(self):
        stream = _Stream(math.nan, guessed=90000, base=Fraction(90000, 1), frames=300)
        self.assertEqual(frames_per_hash(stream, 10.0), 30)  # frames / duration
        self.assertEqual(frames_per_hash(_Stream(math.nan, guessed=0.5, base=None, frames=0), 10.0), 1)

    def test_frames_over_duration(self):
        self.assertEqual(frames_per_hash(_Stream(None, frames=750), 30.0), 25)
        self.assertEqual(frames_per_hash(_Stream(None, frames=750), None), 1)  # no duration: every frame
        self.assertEqual(frames_per_hash(_Stream(None, frames=10_000_000), 1.0), 1)  # absurd estimate

    def test_measured_rate_beats_a_wrong_guess(self):
        # Short variable frame rate WebMs from a real library: guessed_rate 100, really 55 frames in 4.58 s.
        stream = _Stream(_ZeroOverZero(), guessed=100, base=100)
        self.assertEqual(frames_per_hash(stream, 4.58, measured_rate=lambda: 55 / 4.58), 12)
        self.assertEqual(frames_per_hash(stream, 4.58, measured_rate=12.0), 12)
        self.assertEqual(frames_per_hash(stream, 10.0, measured_rate=lambda: 0.2), 1)  # really 0.2 fps: every frame
        self.assertEqual(frames_per_hash(stream, 4.58, measured_rate=lambda: math.nan), 100)  # can't measure: guess
        # A GIF whose header duration is far too short (948 frames in 9.3 s by the header, really 112 frames).
        gif = _Stream(math.inf, guessed=10, base=10, frames=948)
        self.assertEqual(frames_per_hash(gif, 9.296, measured_rate=lambda: 112 / 9.296), 12)

    def test_rate_is_only_measured_when_needed(self):
        def fail():
            raise AssertionError("measured a video with a valid frame rate")

        self.assertEqual(frames_per_hash(_Stream(Fraction(30000, 1001)), 10.0, measured_rate=fail), 30)
        self.assertEqual(frames_per_hash(_Stream(0.5), 10.0, measured_rate=fail), 1)

    def test_nothing_usable_hashes_every_frame(self):
        for average in (math.nan, None, 0):
            self.assertEqual(frames_per_hash(_Stream(average)), 1)

    def test_old_code_failed_on_nan_none_and_inf(self):
        # So those files never had a stored hash that could change.
        for average in (math.nan, None, math.inf):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                _old_frames_per_hash(_Stream(average))


def _write_video(path: Path, codec: str, rate, frames: int, size=(96, 64), pix_fmt="yuv420p"):
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=rate)
        stream.width, stream.height = size
        stream.pix_fmt = pix_fmt
        rng = np.random.default_rng(5)
        base = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        for i in range(frames):
            image = np.roll(base, i * 3, axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TestMeasureFrameRate(unittest.TestCase):
    def test_measures_real_files(self):
        with TemporaryDirectory() as tmp:
            for name, codec, rate, pix_fmt in (("a.mp4", "mpeg4", 30, "yuv420p"), ("b.gif", "gif", 10, "rgb8")):
                path = Path(tmp) / name
                _write_video(path, codec, rate, 40, pix_fmt=pix_fmt)
                self.assertAlmostEqual(vpdqpy.measure_frame_rate(path.read_bytes()), rate, delta=rate * 0.05)
        self.assertTrue(math.isnan(vpdqpy.measure_frame_rate(b"not a video")))


class TestHashesUnchanged(unittest.TestCase):
    """Synthetic videos: the new code gives byte-identical hashes to the old frame rate code."""

    def _compare(self, path: Path):
        new = Vpdq.computeHash(path).bytes
        with mock.patch.object(vpdqpy, "frames_per_hash", lambda stream, **kwargs: _old_frames_per_hash(stream)):
            old = Vpdq.computeHash(path).bytes
        self.assertEqual(new, old)
        self.assertGreater(len(new), 0)
        return new

    def test_normal_video(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "normal.mp4"
            _write_video(path, "mpeg4", Fraction(30000, 1001), 120)
            with av.open(str(path)) as container:
                self.assertGreaterEqual(float(container.streams.video[0].average_rate), 1)
            phash = self._compare(path)
            self.assertLessEqual(len(phash) // 32, 5)  # about one hashed frame per second of a 4 s video

    def test_slow_gif(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "slow.gif"
            _write_video(path, "gif", Fraction(1, 2), 6, pix_fmt="rgb8")
            with av.open(str(path)) as container:
                self.assertEqual(float(container.streams.video[0].average_rate), 0.5)
            self._compare(path)  # every frame, as before


if __name__ == "__main__":
    unittest.main()
