from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.dedup_util import (
    CODE_TOKEN_RE,
    STYLE_CODE,
    STYLE_ERROR_LABEL,
    STYLE_TIP_LABEL,
    STYLE_WARNING_LABEL,
    should_auto_highlight,
    stage_success_text,
    style_user_text,
    user_text,
)


def _span_map(text):
    return [(text.plain[span.start : span.end], str(span.style)) for span in text.spans]


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
            CODE_TOKEN_RE.findall('If you clear or resync potential duplicates in Hydrus ("delete all'),
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
        self.assertTrue(styled.plain.startswith("Speed tips: The final phase of the tool is limited by Hydrus's database queues"))
        self.assertIn(" - Close any Hydrus duplicates pages", styled.plain)
        for snippet, style in _span_map(styled):
            if snippet != "Speed tips:":
                self.fail(f"unexpected style {style!r} on {snippet!r}")

    def test_search_cache_warning_startup_spacing(self):
        # Once at startup: blank line before and after, between filesize and Custom Query.
        with redirect_stdout(io.StringIO()) as out:
            print("Database filesize: 123 bytes.")
            dedup.print_search_cache_warning()
            print("Custom Query: ['tag']")
        printed = out.getvalue().replace("\r\n", "\n")
        self.assertIn("Database filesize: 123 bytes.\n\n", printed)
        self.assertIn("\n\nCustom Query: ['tag']\n", printed)
        # The warning is one paragraph and wraps to the console width.
        printed_flat = " ".join(printed.split())
        warning_flat = " ".join(dedup.SEARCH_CACHE_WARNING.split())
        self.assertIn(warning_flat, printed_flat)
        self.assertNotIn("\n\n\n", printed)

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

    def test_stage_success_line_is_green_with_cyan_counts(self):
        text = stage_success_text("Finished perceptual hashing. Added 2 new perceptual hashes to the database.")
        self.assertEqual(
            text.plain,
            "Finished perceptual hashing. Added 2 new perceptual hashes to the database.",
        )
        self.assertEqual(str(text.style), "green")
        self.assertIn(("2", "repr.number"), _span_map(text))


class TestStageMessages(unittest.TestCase):
    def test_found_videos_to_hash_message(self):
        self.assertEqual(dedup.found_videos_to_hash_message(0), "No new videos to perceptually hash.")
        self.assertEqual(dedup.found_videos_to_hash_message(1), "Found 1 video to perceptually hash.")
        self.assertEqual(dedup.found_videos_to_hash_message(2), "Found 2 videos to perceptually hash.")

    def test_hashing_result_messages(self):
        self.assertEqual(
            dedup.finished_hashing_message(0),
            "Finished perceptual hashing. Added 0 new perceptual hashes to the database.",
        )
        self.assertEqual(
            dedup.finished_hashing_message(1),
            "Finished perceptual hashing. Added 1 new perceptual hash to the database.",
        )
        self.assertEqual(
            dedup.added_hashes_message(2),
            "Added 2 new perceptual hashes to the database.",
        )

    def test_duplicates_result_messages(self):
        self.assertEqual(
            dedup.finished_duplicates_message(0),
            "Finished finding potential duplicates. No new potential duplicate pairs found.",
        )
        self.assertEqual(
            dedup.finished_duplicates_message(1),
            "Finished finding potential duplicates. 1 new potential duplicate pair marked for manual processing.",
        )
        self.assertEqual(
            dedup.finished_duplicates_message(5),
            "Finished finding potential duplicates. 5 new potential duplicate pairs marked for manual processing.",
        )
        self.assertEqual(
            dedup.duplicates_result_message(0),
            "No new potential duplicate pairs found.",
        )
        self.assertEqual(
            dedup.duplicates_result_message(1),
            "1 new potential duplicate pair marked for manual processing.",
        )


if __name__ == "__main__":
    unittest.main()
