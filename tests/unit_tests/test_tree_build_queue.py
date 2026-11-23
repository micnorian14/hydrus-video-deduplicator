from __future__ import annotations

import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hydrusvideodeduplicator import db_tools, dedup, entrypoint
from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes
from .test_orphans import _FakeClient, _FakeDeduplicator
from .test_relink import _new_db, _tables

BASE = 20  # videos already in the search tree; the rest of the hashes are queued


def _hashes() -> list[bytes]:
    return [h for h in _make_hashes(seed=41) if h] + [h for h in _make_hashes(seed=42) if h]


def _file_hash(i: int) -> str:
    return f"{i:064x}"


def _make(tmpdir: str, name: str) -> tuple[DedupeDB.DedupeDb, list[tuple[str, bytes]]]:
    hashes = _hashes()
    db = _new_db(tmpdir, name, hashes[:BASE])
    queued = [(_file_hash(i), phash) for i, phash in enumerate(hashes[BASE:], BASE)]
    for file_hash, phash in queued:
        db.add_to_phashed_files_queue(file_hash, phash)
    db.commit()
    return db, queued


def _open(path: Path) -> DedupeDB.DedupeDb:
    vptree.TemporaryIntegerTableNameCache()
    db = DedupeDB.DedupeDb(path.parent, path.name)
    db.init_connection()
    return db


def _build_tree(db: DedupeDB.DedupeDb) -> str:
    """Run process_phashed_file_queue() exactly the way dedup.py does (Ctrl+C is caught inside `with db.conn:`)."""
    d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
    d.db = db
    d.update_progress_callback = None
    d.should_skip_step_fn = None
    db.begin_transaction()
    with db.conn:
        try:
            d.process_phashed_file_queue()
        except KeyboardInterrupt:
            return "interrupted"
    return "finished"


def _interrupt_on(call: int, after_real: bool):
    """Patch for associate_file_with_perceptual_hash: Ctrl+C on the given call, before or after its tree writes."""
    real = DedupeDB.DedupeDb.associate_file_with_perceptual_hash
    calls = [0]

    def patched(self, file_hash, perceptual_hash):
        calls[0] += 1
        if calls[0] == call:
            if after_real:
                real(self, file_hash, perceptual_hash)
            raise KeyboardInterrupt
        return real(self, file_hash, perceptual_hash)

    return mock.patch.object(DedupeDB.DedupeDb, "associate_file_with_perceptual_hash", patched)


class TestTreeBuildCtrlC(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _reference(self, tmpdir: str) -> dict:
        db, _ = _make(tmpdir, "reference.sqlite")
        self.assertEqual(_build_tree(db), "finished")
        out = _tables(db)
        db.close()
        return out

    def _check_ctrl_c(self, fast: bool, call: int, after_real: bool):
        with mock.patch.dict(os.environ, {"HVD_FAST_INSERT": "1" if fast else "0"}), TemporaryDirectory() as tmpdir:
            expected = self._reference(tmpdir)
            vptree.TemporaryIntegerTableNameCache()
            db, queued = _make(tmpdir, "interrupted.sqlite")
            self.assertGreater(len(queued), call)
            with _interrupt_on(call, after_real):
                self.assertEqual(_build_tree(db), "interrupted")
            db.close()

            # What was committed: exactly the videos finished before the Ctrl+C, nothing of the unfinished one.
            path = Path(tmpdir) / "interrupted.sqlite"
            db = _open(path)
            remaining = [row[0] for row in db.execute("SELECT file_hash FROM phashed_file_queue;").fetchall()]
            self.assertEqual(sorted(remaining), sorted(h for h, _ in queued[call - 1 :]))
            self.assertEqual(db.count_files(), BASE + call - 1)
            self.assertIsNone(db.get_hash_id(queued[call - 1][0]))  # no half-added file row
            self.assertEqual(db.count_orphaned_files(), 0)
            self.assertEqual(db.check_search_tree()["problems"], 0)
            self.assertEqual(db.count_queued_files(), len(queued) - call + 1)

            # The next run finishes the rest; the result is the same as one uninterrupted run.
            self.assertEqual(_build_tree(db), "finished")
            self.assertEqual(_tables(db), expected)
            db.close()

    def test_ctrl_c_before_tree_insert_fast(self):
        self._check_ctrl_c(fast=True, call=5, after_real=False)

    def test_ctrl_c_after_tree_insert_fast(self):
        self._check_ctrl_c(fast=True, call=5, after_real=True)

    def test_ctrl_c_after_a_periodic_commit_fast(self):
        self._check_ctrl_c(fast=True, call=70, after_real=True)

    def test_ctrl_c_original_insert_code(self):
        self._check_ctrl_c(fast=False, call=70, after_real=True)

    def test_half_added_video_from_an_older_version_is_finished(self):
        # Older versions could commit a queued video's file row and perceptual hash without its map, tree node or
        # search cache entry, leaving it in the queue. The next run must finish it like any other queued video.
        with TemporaryDirectory() as tmpdir:
            expected = self._reference(tmpdir)
            vptree.TemporaryIntegerTableNameCache()
            db, queued = _make(tmpdir, "legacy.sqlite")
            for file_hash, phash in queued[:2]:
                db.add_file(file_hash)
                db.add_perceptual_hash(phash)
            db.commit()
            hash_id = db.get_hash_id(queued[0][0])
            self.assertIsNotNone(hash_id)
            self.assertIsNone(
                db.execute("SELECT 1 FROM shape_perceptual_hash_map WHERE hash_id = ?;", (hash_id,)).fetchone()
            )

            self.assertEqual(_build_tree(db), "finished")
            self.assertEqual(db.count_queued_files(), 0)
            self.assertEqual(db.count_orphaned_files(), 0)
            self.assertEqual(db.check_search_tree()["problems"], 0)
            self.assertEqual(_tables(db), expected)
            db.close()


class TestQueueWaitingLine(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def test_message(self):
        self.assertIsNone(db_tools.queue_waiting_message(0))
        self.assertEqual(
            db_tools.queue_waiting_message(1), "1 hashed video is still waiting to be added to the search tree."
        )
        self.assertEqual(
            db_tools.queue_waiting_message(289), "289 hashed videos are still waiting to be added to the search tree."
        )

    def test_printed_only_when_the_queue_is_not_empty(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "queue.sqlite", _make_hashes(seed=43)[:5])
            with mock.patch.object(db_tools, "print_and_log") as printed:
                self.assertEqual(db_tools.print_queue_waiting(db), 0)
            printed.assert_not_called()

            db.add_to_phashed_files_queue("f" * 64, b"\x01" * 32)
            db.commit()
            with mock.patch.object(db_tools, "print_and_log") as printed:
                self.assertEqual(db_tools.print_queue_waiting(db), 1)
            printed.assert_called_once()
            self.assertEqual(printed.call_args.args[1], db_tools.queue_waiting_message(1))
            self.assertEqual(len(printed.call_args.args), 2)  # default (info) style, like the other stats lines
            self.assertFalse(db.conn.in_transaction)  # read only
            db.close()

    def test_cli_prints_it_after_the_health_lines(self):
        old_db_dir = DedupeDB.get_db_dir()
        try:
            with TemporaryDirectory() as tmpdir:
                folder = Path(tmpdir)
                db = _new_db(tmpdir, "videohashes.sqlite", _make_hashes(seed=44)[:5])
                db.add_to_phashed_files_queue("f" * 64, b"\x01" * 32)
                db.add_to_phashed_files_queue("e" * 64, b"\x02" * 32)
                db.commit()
                db.close()
                order = []
                with (
                    mock.patch.object(entrypoint, "create_client", return_value=_FakeClient()),
                    mock.patch.object(entrypoint, "HydrusVideoDeduplicator", _FakeDeduplicator),
                    mock.patch.object(entrypoint, "check_database", side_effect=lambda *a, **k: order.append("check")),
                    mock.patch.object(db_tools, "print_and_log", side_effect=lambda log, msg: order.append(msg)),
                    mock.patch.object(
                        entrypoint, "print_search_cache_warning", side_effect=lambda: order.append("warning")
                    ),
                ):
                    entrypoint.main(api_key="x" * 64, api_url="http://127.0.0.1:1", dedup_database_dir=folder)
                self.assertEqual(order, ["check", db_tools.queue_waiting_message(2), "warning"])
        finally:
            DedupeDB.set_db_dir(old_db_dir)
            logging.disable(logging.NOTSET)  # main() disables logging when not verbose


if __name__ == "__main__":
    unittest.main()
