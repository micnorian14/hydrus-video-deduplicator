from __future__ import annotations

import unittest

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.dedup_util import (
    CODE_TOKEN_RE,
    STYLE_CODE,
    STYLE_ERROR_LABEL,
    STYLE_TIP_LABEL,
    STYLE_WARNING_LABEL,
    should_auto_highlight,
    style_user_text,
    user_text,
)


def _span_map(text):
    return [(text.plain[span.start:span.end], str(span.style)) for span in text.spans]


class TestConsoleStyle(unittest.TestCase):
    def test_code_tokens(self):
        self.assertEqual(CODE_TOKEN_RE.findall("press CTRL+C now"), ["CTRL+C"])
        self.assertEqual(CODE_TOKEN_RE.findall("(--clear-search-cache)"), ["--clear-search-cache"])
        self.assertEqual(
            CODE_TOKEN_RE.findall("with '--failed-page-name=failed'"),
            ["--failed-page-name=failed"],
        )
        self.assertEqual(CODE_TOKEN_RE.findall("'--clear-search-tree'."), ["--clear-search-tree"])
        self.assertEqual(
            CODE_TOKEN_RE.findall("HVD_SKIP_REPEAT_PAIRS=1 or --api-key"),
            ["HVD_SKIP_REPEAT_PAIRS=1", "--api-key"],
        )
        # Quoted Hydrus menu item names are not flags
        self.assertEqual(
            CODE_TOKEN_RE.findall(
                'If you clear or resync potential duplicates in Hydrus ("delete all'
            ),
            [],
        )

    def test_tip_label_and_ctrl_c(self):
        raw = "\nTip: You can skip perceptual hashing at any time by pressing CTRL+C.\n"
        styled = style_user_text(raw)
        self.assertEqual(styled.plain, raw)
        self.assertIn(("Tip:", STYLE_TIP_LABEL), _span_map(styled))
        self.assertIn(("CTRL+C", STYLE_CODE), _span_map(styled))

    def test_speed_tips_only_label_is_colored(self):
        styled = style_user_text(dedup.SPEED_TIPS)
        self.assertEqual(styled.plain, dedup.SPEED_TIPS)
        self.assertIn(("Speed tips:", STYLE_TIP_LABEL), _span_map(styled))
        self.assertTrue(styled.plain.startswith("Speed tips: This step is limited by Hydrus's database"))
        self.assertIn(" - Close any Hydrus duplicates pages", styled.plain)
        for snippet, style in _span_map(styled):
            if snippet != "Speed tips:":
                self.fail(f"unexpected style {style!r} on {snippet!r}")

    def test_search_cache_warning_colors(self):
        styled = style_user_text(dedup.SEARCH_CACHE_WARNING)
        self.assertEqual(styled.plain, dedup.SEARCH_CACHE_WARNING)
        self.assertIn(("WARNING:", STYLE_WARNING_LABEL), _span_map(styled))
        self.assertEqual(STYLE_WARNING_LABEL, "red3")
        self.assertNotIn("bold", STYLE_WARNING_LABEL)
        self.assertIn(("--clear-search-cache", STYLE_CODE), _span_map(styled))
        # Only the label and the flag are colored; the rest, including quoted Hydrus menu names, stays plain.
        styled_map = dict(_span_map(styled))
        self.assertEqual(set(styled_map), {"WARNING:", "--clear-search-cache"})
        self.assertNotIn("delete all potential duplicate pairs and re-search", styled_map)

    def test_error_prefix_and_literal_brackets(self):
        raw = "ERROR clearing search tree while to get file_hash from hash_id 1. file_hash not found."
        styled = style_user_text(raw)
        self.assertEqual(styled.plain, raw)
        self.assertIn(("ERROR", STYLE_ERROR_LABEL), _span_map(styled))
        exc = "[Errno 129] Error number -129 occurred"
        styled_exc = style_user_text(exc)
        self.assertEqual(styled_exc.plain, exc)



    def test_startup_lines_keep_rich_highlighting(self):
        # These unlabeled print_and_log startup lines used to get Rich auto-highlighting
        # (URL blue/underline, quoted strings green, numbers cyan). user_text must restore that.
        cases = [
            ("Connecting to Hydrus at http://localhost:45869", "http://localhost:45869", "repr.url"),
            ("Dedupe API version: 'v70'", "'v70'", "repr.str"),
            ("Hydrus API version: 'v96'", "'v96'", "repr.str"),
            (
                "Found existing database at: '%USERPROFILE%\\AppData\\Local\\videohashes.sqlite'",
                "'%USERPROFILE%\\AppData\\Local\\videohashes.sqlite'",
                "repr.str",
            ),
            ("Database has 410776 videos already perceptually hashed.", "410776", "repr.number"),
            ("Database filesize: 773132288 bytes.", "773132288", "repr.number"),
        ]
        for raw, snippet, style in cases:
            with self.subTest(raw=raw):
                self.assertTrue(should_auto_highlight(raw))
                styled = user_text(raw)
                self.assertEqual(styled.plain, raw)
                self.assertIn((snippet, style), _span_map(styled))

        # Labeled lines and lines with flags stay protected (no ReprHighlighter).
        tip = "\nTip: You can skip perceptual hashing at any time by pressing CTRL+C.\n"
        self.assertFalse(should_auto_highlight(tip))
        self.assertIn(("Tip:", STYLE_TIP_LABEL), _span_map(user_text(tip)))
        self.assertIn(("CTRL+C", STYLE_CODE), _span_map(user_text(tip)))
        warning = dedup.SEARCH_CACHE_WARNING
        self.assertFalse(should_auto_highlight(warning))

        # Literal error text can opt out of highlighting; brackets stay characters.
        exc = "[Errno 129] Error number -129 occurred"
        styled_exc = user_text(exc, highlight=False)
        self.assertEqual(styled_exc.plain, exc)
        self.assertNotIn("repr.brace", [style for _, style in _span_map(styled_exc)])


if __name__ == "__main__":
    unittest.main()
