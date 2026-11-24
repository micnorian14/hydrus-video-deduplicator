"""--dry-run: write found pairs to a file, send nothing to Hydrus, leave the search cache unchanged."""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.page_logger import HydrusPageLogger

from .test_pipeline import THRESHOLD, Recorder, _build, _open, _run


def _cache_rows(path: Path) -> list[tuple]:
    db = _open(path)
    rows = db.execute("SELECT hash_id, searched_distance FROM shape_search_cache ORDER BY hash_id;").fetchall()
    db.close()
    return rows


def _read_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_dry(path: Path) -> tuple[object, object, MagicMock]:
    """Run find_potential_duplicates() in dry-run, with a client that fails on any Hydrus write."""
    old_pipeline = os.environ.get("HVD_PIPELINE")
    # Pipeline on: dry-run must still skip Hydrus writes and use the file path.
    os.environ["HVD_PIPELINE"] = "1"
    db = _open(path)
    d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
    d.db = db
    d.similarity_threshold = THRESHOLD
    d.update_progress_callback = None
    d.should_skip_step_fn = None
    d.dry_run = True
    d.skip_repeat_pairs = False
    client = MagicMock()
    client.client.set_file_relationships.side_effect = AssertionError("set_file_relationships must not be called")
    client.client.add_files_to_page.side_effect = AssertionError("add_files_to_page must not be called")
    client.client.remove_potentials.side_effect = AssertionError("remove_potentials must not be called")
    d.client = client
    try:
        db.begin_transaction()
        with db.conn:
            result = d.find_potential_duplicates()
        return result, d, client
    finally:
        db.close()
        if old_pipeline is None:
            os.environ.pop("HVD_PIPELINE", None)
        else:
            os.environ["HVD_PIPELINE"] = old_pipeline


class TestDryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name) / "base.sqlite"
        _build(self.base)
        self._old_dry = os.environ.get("HVD_DRY_RUN")
        os.environ.pop("HVD_DRY_RUN", None)

    def tearDown(self):
        self.tmp.cleanup()
        if self._old_dry is None:
            os.environ.pop("HVD_DRY_RUN", None)
        else:
            os.environ["HVD_DRY_RUN"] = self._old_dry

    def _copy(self, name: str) -> Path:
        dst = Path(self.tmp.name) / name
        shutil.copyfile(self.base, dst)
        return dst

    def test_similarity_from_search_distance(self):
        self.assertEqual(dedup.similarity_from_search_distance(1), 100.0)
        self.assertEqual(dedup.similarity_from_search_distance(51), 50.0)
        self.assertEqual(dedup.similarity_from_search_distance(101), 0.0)
        self.assertIsNone(dedup.similarity_from_search_distance(None))

    def test_writer_scores_both_directions(self):
        path = Path(self.tmp.name) / "dry_run_pairs-20261122-000000.jsonl"
        writer = dedup.DryRunPairWriter(path)
        try:
            writer.record("a" * 64, "b" * 64, distance=11)
            writer.record("b" * 64, "a" * 64, distance=21)
            writer.record("c" * 64, "d" * 64, distance=None)
        finally:
            writer.close()
        rows = _read_pairs(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0],
            {
                "file_a": "a" * 64,
                "file_b": "b" * 64,
                "score_a_to_b": 90.0,
                "score_b_to_a": 80.0,
            },
        )
        self.assertEqual(
            rows[1],
            {
                "file_a": "c" * 64,
                "file_b": "d" * 64,
                "score_a_to_b": None,
                "score_b_to_a": None,
            },
        )

    def test_writer_unmeasured_reverse_score_is_null(self):
        path = Path(self.tmp.name) / "dry_run_pairs-oneway.jsonl"
        writer = dedup.DryRunPairWriter(path)
        try:
            writer.record("a" * 64, "b" * 64, distance=1)
            rows = _read_pairs(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["score_a_to_b"], 100.0)
            self.assertIsNone(rows[0]["score_b_to_a"])

            writer.record("b" * 64, "a" * 64, distance=76)
        finally:
            writer.close()
        self.assertEqual(
            _read_pairs(path),
            [
                {
                    "file_a": "a" * 64,
                    "file_b": "b" * 64,
                    "score_a_to_b": 100.0,
                    "score_b_to_a": 25.0,
                }
            ],
        )

    def test_writer_forward_fill_does_not_copy_into_reverse(self):
        path = Path(self.tmp.name) / "dry_run_pairs-forward-fill.jsonl"
        writer = dedup.DryRunPairWriter(path)
        try:
            writer.record("a" * 64, "b" * 64, distance=None)
            writer.record("a" * 64, "b" * 64, distance=1)
        finally:
            writer.close()
        rows = _read_pairs(path)
        self.assertEqual(rows[0]["score_a_to_b"], 100.0)
        self.assertIsNone(rows[0]["score_b_to_a"])

    def test_writes_pairs_file_and_does_not_call_hydrus(self):
        path = self._copy("dry.sqlite")
        result, d, client = _run_dry(path)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
        client.client.set_file_relationships.assert_not_called()
        client.client.add_files_to_page.assert_not_called()
        client.client.remove_potentials.assert_not_called()

        jsonl_paths = list(path.parent.glob("dry_run_pairs-*.jsonl"))
        self.assertEqual(len(jsonl_paths), 1)
        self.assertEqual(jsonl_paths[0], d.dry_run_pairs_path)
        data = _read_pairs(jsonl_paths[0])
        self.assertGreater(len(data), 0)
        self.assertEqual(len(data), result)
        seen = set()
        for row in data:
            self.assertEqual(set(row), {"file_a", "file_b", "score_a_to_b", "score_b_to_a"})
            self.assertEqual(len(row["file_a"]), 64)
            self.assertEqual(len(row["file_b"]), 64)
            self.assertNotEqual(row["file_a"], row["file_b"])
            for key in ("score_a_to_b", "score_b_to_a"):
                score = row[key]
                self.assertTrue(score is None or (isinstance(score, (int, float)) and 0 <= score <= 100))
            key = tuple(sorted((row["file_a"], row["file_b"])))
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_search_cache_unchanged_so_later_run_still_sends(self):
        dry_path = self._copy("cache-dry.sqlite")
        before = _cache_rows(dry_path)
        self.assertTrue(all(distance is None for _, distance in before))

        result, _, client = _run_dry(dry_path)
        self.assertGreater(result, 0)
        client.client.set_file_relationships.assert_not_called()
        self.assertEqual(_cache_rows(dry_path), before)

        rec = Recorder(delay=0)
        after = _run(dry_path, False, rec)
        self.assertGreater(len(rec.calls), 0)

        fresh = self._copy("cache-fresh.sqlite")
        fresh_rec = Recorder(delay=0)
        fresh_result = _run(fresh, False, fresh_rec)
        self.assertEqual(fresh_result, after)
        self.assertEqual(fresh_rec.calls, rec.calls)

    def test_mark_videos_as_duplicates_does_not_call_hydrus(self):
        path = self._copy("mark.sqlite")
        db = _open(path)
        d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        d.db = db
        d.dry_run = True
        client = MagicMock()
        client.client.set_file_relationships.side_effect = AssertionError("set_file_relationships must not be called")
        d.client = client
        try:
            d.mark_videos_as_duplicates("a" * 64, "b" * 64, distance=4)
            client.client.set_file_relationships.assert_not_called()
            jsonl_paths = list(path.parent.glob("dry_run_pairs-*.jsonl"))
            self.assertEqual(len(jsonl_paths), 1)
            rows = _read_pairs(jsonl_paths[0])
            self.assertEqual(
                rows,
                [
                    {
                        "file_a": "a" * 64,
                        "file_b": "b" * 64,
                        "score_a_to_b": 97.0,
                        "score_b_to_a": None,
                    }
                ],
            )
        finally:
            d._close_dry_run_writer()
            db.close()

    def test_page_logger_sends_nothing(self):
        client = MagicMock()
        logger = HydrusPageLogger(client, "failed", dry_run=True)
        logger.add_failed_video("abcd")
        client.client.get_pages.assert_not_called()
        client.client.add_files_to_page.assert_not_called()
        self.assertEqual(logger.sent, 0)

    def test_env_var_enables_dry_run(self):
        os.environ["HVD_DRY_RUN"] = "1"
        self.assertTrue(dedup.dry_run_enabled_by_env())
        path = self._copy("env.sqlite")
        old_pipeline = os.environ.get("HVD_PIPELINE")
        os.environ["HVD_PIPELINE"] = "0"
        db = _open(path)
        d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        d.db = db
        d.similarity_threshold = THRESHOLD
        d.update_progress_callback = None
        d.should_skip_step_fn = None
        d.dry_run = False
        d.skip_repeat_pairs = False
        client = MagicMock()
        client.client.set_file_relationships.side_effect = AssertionError("set_file_relationships must not be called")
        d.client = client
        try:
            db.begin_transaction()
            with db.conn:
                result = d.find_potential_duplicates()
            self.assertGreater(result, 0)
            client.client.set_file_relationships.assert_not_called()
            self.assertTrue(list(path.parent.glob("dry_run_pairs-*.jsonl")))
        finally:
            db.close()
            if old_pipeline is None:
                os.environ.pop("HVD_PIPELINE", None)
            else:
                os.environ["HVD_PIPELINE"] = old_pipeline

    def test_help_lists_dry_run(self):
        import inspect

        from hydrusvideodeduplicator.entrypoint import main

        source = inspect.getsource(main)
        self.assertIn("dry_run", source)
        self.assertIn("Don't send anything to Hydrus", source)


if __name__ == "__main__":
    unittest.main()
