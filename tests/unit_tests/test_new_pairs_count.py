"""The new-pair summary uses pairs found this run, not Hydrus's changing total."""

from __future__ import annotations

import unittest
from unittest import mock

from hydrusvideodeduplicator import dedup


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Db:
    def begin_transaction(self):
        return None

    conn = _Conn()


class TestNewPairsCount(unittest.TestCase):
    def _run_body(self, found_pairs: int, hydrus_before: int, hydrus_after: int) -> list[str]:
        d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        d.dry_run = False
        d.gpu_decode_auto = False
        d.gpu_profile = None
        d.update_progress_callback = None
        d.hydlog = mock.Mock()
        d.db = _Db()
        d.client = mock.Mock()
        # A falling Hydrus total would have printed a negative number with the old subtraction.
        d.client.get_potential_duplicate_count_hydrus.side_effect = [hydrus_before, hydrus_after]
        d.process_phashed_file_queue = mock.Mock()
        d.run_maintenance = mock.Mock()
        d.find_potential_duplicates = mock.Mock(return_value=found_pairs)
        messages: list[str] = []
        with mock.patch.object(dedup, "print_success", side_effect=lambda s: messages.append(s)):
            with mock.patch.object(dedup, "print_user"):
                result = d._deduplicate_body(skip_hashing=True)
        self.assertEqual(result, found_pairs)
        return messages

    def test_zero_found_is_not_negative_when_hydrus_count_drops(self):
        messages = self._run_body(found_pairs=0, hydrus_before=20, hydrus_after=0)
        self.assertIn(
            "Finished finding potential duplicates. No new potential duplicate pairs found.",
            messages,
        )
        for message in messages:
            self.assertNotRegex(message, r"-\d+ new potential duplicate")

    def test_uses_pairs_found_this_run_not_hydrus_delta(self):
        messages = self._run_body(found_pairs=3, hydrus_before=20, hydrus_after=0)
        self.assertIn(
            "Finished finding potential duplicates. 3 new potential duplicate pairs marked for manual processing.",
            messages,
        )
        self.assertNotIn(
            "Finished finding potential duplicates. -17 new potential duplicate pairs marked for manual processing.",
            messages,
        )


if __name__ == "__main__":
    unittest.main()
