from __future__ import annotations

import io
import logging
import unittest

from tqdm.utils import disp_len

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.dedup_util import TqdmLoggingHandler


class TestProgressBarWidth(unittest.TestCase):
    def test_capped_bar_cols(self):
        self.assertEqual(dedup.MAX_BAR_COLS, 110)
        self.assertEqual(dedup.capped_bar_cols(250), 110)
        self.assertEqual(dedup.capped_bar_cols(110), 110)
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
        self.assertEqual(self._render(200), 110)
        self.assertEqual(self._render(60), 60)

    def test_dynamic_width_is_capped_each_refresh(self):
        out = io.StringIO()
        bar = dedup.ProgressBar(total=10, file=out, dynamic_ncols=True)
        widths = [150, 70, 300]
        bar.dynamic_ncols = lambda fp: (widths.pop(0) if widths else 80, 24)  # simulate the window being resized
        got = [bar.format_dict["ncols"] for _ in range(3)]
        bar.close()
        self.assertEqual(got, [110, 70, 110])

    def test_same_arguments_as_tqdm(self):
        bar = dedup.ProgressBar(total=5, file=io.StringIO(), dynamic_ncols=True, smoothing=dedup.TQDM_SMOOTHING,
                                colour="GREEN", unit="file", desc="x")
        self.assertEqual((bar.smoothing, bar.colour, bar.unit, bar.desc), (dedup.TQDM_SMOOTHING, "GREEN", "file", "x"))
        bar.close()

    @staticmethod
    def _visible_lines(s: str) -> list[str]:
        # tqdm uses carriage return to clear and redraw; apply those overwrites like a terminal would.
        lines = []
        for line in s.split("\n"):
            cur = ""
            for part in line.split("\r"):
                cur = part + cur[len(part):]
            lines.append(cur)
        return lines

    def test_messages_print_on_their_own_line(self):
        # Mid-bar prints used to land on the bar's line and wrap. external_write_mode
        # (used for hashing failure messages) clears the bar, writes the message, then redraws.
        out = io.StringIO()
        bar = dedup.ProgressBar(total=2, file=out, ncols=50, desc="Perceptually hashing files", unit="file",
                                mininterval=0, miniters=1, leave=True, smoothing=0)
        bar.refresh()
        with bar.external_write_mode(file=bar.fp):
            bar.fp.write("Failed to perceptually hash a file.\n")
        lines = [line.rstrip() for line in self._visible_lines(out.getvalue())]
        bar.close()
        self.assertIn("Failed to perceptually hash a file.", lines)
        # The message must not be glued to the end of a bar line.
        for line in lines:
            if "Failed to perceptually hash a file." in line:
                self.assertEqual(line, "Failed to perceptually hash a file.")
        # The bar should be redrawn after the message.
        idx = lines.index("Failed to perceptually hash a file.")
        self.assertTrue(any("Perceptually hashing files" in line for line in lines[idx + 1:]))

    def test_verbose_log_prints_on_their_own_line(self):
        # hydlog.info (e.g. 'Similar files found') goes to the console under --verbose.
        # TqdmLoggingHandler routes those through tqdm.write so they don't append to the live bar.
        out = io.StringIO()
        bar = dedup.ProgressBar(total=2, file=out, ncols=50, desc="Finding potential duplicates", unit="file",
                                mininterval=0, miniters=1, leave=True, smoothing=0)
        bar.refresh()
        handler = TqdmLoggingHandler(stream=bar.fp)
        handler.setFormatter(logging.Formatter(" %(asctime)s - %(name)s: %(message)s", "%H:%M:%S"))
        logger = logging.getLogger("hvd_tqdm_handler_test")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info('Similar files found: "aaa" and "bbb"')
        lines = [line.rstrip() for line in self._visible_lines(out.getvalue())]
        bar.close()
        found = [line for line in lines if "Similar files found" in line]
        self.assertEqual(len(found), 1)
        self.assertIn('Similar files found: "aaa" and "bbb"', found[0])
        self.assertNotIn("Finding potential duplicates", found[0])
        idx = lines.index(found[0])
        self.assertTrue(any("Finding potential duplicates" in line for line in lines[idx + 1:]))


if __name__ == "__main__":
    unittest.main()
