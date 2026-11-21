from __future__ import annotations

import io
import unittest

from tqdm.utils import disp_len

from hydrusvideodeduplicator import dedup


class TestProgressBarWidth(unittest.TestCase):
    def test_capped_bar_cols(self):
        self.assertEqual(dedup.MAX_BAR_COLS, 100)
        self.assertEqual(dedup.capped_bar_cols(250), 100)
        self.assertEqual(dedup.capped_bar_cols(100), 100)
        self.assertEqual(dedup.capped_bar_cols(79), 79)  # narrower windows still shrink the bar
        self.assertEqual(dedup.capped_bar_cols(0), 1)
        self.assertIsNone(dedup.capped_bar_cols(None))  # unknown width: left to tqdm

    def _render(self, cols):
        out = io.StringIO()
        bar = dedup.ProgressBar(total=1000, file=out, ncols=cols, desc="Finding potential duplicates", unit="file",
                                colour="GREEN", smoothing=dedup.TQDM_SMOOTHING)
        bar.update(123)
        text = str(bar)
        bar.close()
        return disp_len(text)  # visible width, without the colour codes

    def test_bar_never_wider_than_cap(self):
        self.assertEqual(self._render(200), 100)
        self.assertEqual(self._render(60), 60)

    def test_dynamic_width_is_capped_each_refresh(self):
        out = io.StringIO()
        bar = dedup.ProgressBar(total=10, file=out, dynamic_ncols=True)
        widths = [150, 70, 300]
        bar.dynamic_ncols = lambda fp: (widths.pop(0) if widths else 80, 24)  # simulate the window being resized
        got = [bar.format_dict["ncols"] for _ in range(3)]
        bar.close()
        self.assertEqual(got, [100, 70, 100])

    def test_same_arguments_as_tqdm(self):
        bar = dedup.ProgressBar(total=5, file=io.StringIO(), dynamic_ncols=True, smoothing=dedup.TQDM_SMOOTHING,
                                colour="GREEN", unit="file", desc="x")
        self.assertEqual((bar.smoothing, bar.colour, bar.unit, bar.desc), (dedup.TQDM_SMOOTHING, "GREEN", "file", "x"))
        bar.close()


if __name__ == "__main__":
    unittest.main()
