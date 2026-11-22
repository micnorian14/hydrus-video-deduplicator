from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.page_logger import (
    DEFAULT_FAILED_PAGE_NAME,
    HydrusPageLogger,
    resolve_failed_page_name,
)


def _pages(name: str | None = None, page_key: str = "page-key", page_type: int = 6) -> dict:
    root = {"name": "top", "page_type": 0, "pages": []}
    if name is not None:
        root["pages"].append({"name": name, "page_type": page_type, "page_key": page_key})
    return {"pages": root}


def _logger(pages=None, get_pages=None, add_files=None) -> tuple[HydrusPageLogger, mock.Mock, mock.Mock]:
    client = mock.Mock()
    client.client.get_pages.side_effect = get_pages if get_pages is not None else (lambda: pages)
    if add_files is not None:
        client.client.add_files_to_page.side_effect = add_files
    return HydrusPageLogger(client, "failed"), client.client.get_pages, client.client.add_files_to_page


class TestResolveFailedPageName(unittest.TestCase):
    def test_default(self):
        self.assertEqual(DEFAULT_FAILED_PAGE_NAME, "failed")
        self.assertEqual(resolve_failed_page_name(None), "failed")
        self.assertEqual(resolve_failed_page_name(""), "failed")
        self.assertEqual(resolve_failed_page_name("   "), "failed")

    def test_custom_name_is_kept(self):
        self.assertEqual(resolve_failed_page_name("broken"), "broken")
        self.assertEqual(resolve_failed_page_name("  Broken Videos  "), "Broken Videos")


class TestFailedPageMessages(unittest.TestCase):
    def test_sent(self):
        self.assertEqual(
            dedup.failed_page_sent_message(1, "failed"),
            "Sent 1 failed file to the Hydrus page 'failed'.",
        )
        self.assertEqual(
            dedup.failed_page_sent_message(2, "broken"),
            "Sent 2 failed files to the Hydrus page 'broken'.",
        )

    def test_missing_tip_uses_the_page_name(self):
        self.assertEqual(
            dedup.failed_page_missing_tip("failed"),
            "Tip: Create a Hydrus page with the name 'failed' before running the program "
            "to see any files that failed to hash.",
        )
        self.assertIn("'broken'", dedup.failed_page_missing_tip("broken"))


class TestHydrusPageLogger(unittest.TestCase):
    def test_missing_page_is_quiet_and_looks_up_once(self):
        logger, get_pages, add_files = _logger(pages=_pages(None))
        with redirect_stdout(io.StringIO()) as out:
            logger.add_failed_video("a" * 64)
            logger.add_failed_video("b" * 64)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(get_pages.call_count, 1)
        add_files.assert_not_called()
        self.assertTrue(logger.page_missing)
        self.assertEqual(logger.sent, 0)

    def test_existing_page_caches_the_key_and_sends_each_file(self):
        logger, get_pages, add_files = _logger(pages=_pages("failed", page_key="k1"))
        with redirect_stdout(io.StringIO()) as out:
            logger.add_failed_video("aaa")
            logger.add_failed_video("bbb")
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(get_pages.call_count, 1)
        self.assertEqual(
            add_files.call_args_list,
            [
                mock.call(page_key="k1", hashes=["aaa"]),
                mock.call(page_key="k1", hashes=["bbb"]),
            ],
        )
        self.assertFalse(logger.page_missing)
        self.assertEqual(logger.sent, 2)

    def test_lookup_error_is_quiet_and_not_retried(self):
        logger, get_pages, add_files = _logger(get_pages=mock.Mock(side_effect=RuntimeError("hydrus down")))
        with redirect_stdout(io.StringIO()) as out:
            logger.add_failed_video("aaa")
            logger.add_failed_video("bbb")
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(get_pages.call_count, 1)
        add_files.assert_not_called()
        self.assertFalse(logger.page_missing)
        self.assertEqual(logger.sent, 0)

    def test_add_error_is_quiet_and_does_not_count_as_sent(self):
        logger, get_pages, add_files = _logger(
            pages=_pages("failed", page_key="k1"),
            add_files=mock.Mock(side_effect=RuntimeError("page closed")),
        )
        with redirect_stdout(io.StringIO()) as out:
            logger.add_failed_video("aaa")
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(get_pages.call_count, 1)
        self.assertFalse(logger.page_missing)
        self.assertEqual(logger.sent, 0)


if __name__ == "__main__":
    unittest.main()
