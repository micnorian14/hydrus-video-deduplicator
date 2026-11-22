from __future__ import annotations

import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hydrusvideodeduplicator import db_tools, entrypoint
from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes
from .test_relink import _new_db, _tables


def _break(db: DedupeDB.DedupeDb) -> dict:
    """Add the kinds of rows found in a real database: files whose perceptual hash row is missing, and the empty
    perceptual hash (valid: a video with no usable frames). Also marks the valid files as searched."""
    db.execute("UPDATE shape_search_cache SET searched_distance = 51;")
    db.add_file("e" * 64)
    db.add_perceptual_hash(b"")
    db.associate_file_with_perceptual_hash("e" * 64, b"")
    broken = []
    for name, phash_id in (("a", 900001), ("b", 900002), ("c", 900002)):  # two files share one missing hash
        db.execute("INSERT INTO files ( file_hash ) VALUES ( ? );", (name * 64,))
        hash_id = db.get_hash_id(name * 64)
        db.execute("INSERT INTO shape_perceptual_hash_map ( phash_id, hash_id ) VALUES ( ?, ? );", (phash_id, hash_id))
        broken.append(hash_id)
    db.execute("INSERT INTO shape_search_cache ( hash_id, searched_distance ) VALUES ( ?, 51 );", (broken[0],))
    db.commit()
    return {"broken": broken, "empty_phash_id": db.get_phash_id(b"")}


def _without(rows: list, hash_ids: list[int], column: int) -> list:
    return [row for row in rows if row[column] not in hash_ids]


class TestOrphanedFiles(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def test_count_and_remove(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "orphans.sqlite", _make_hashes(seed=3))
            self.assertEqual(db.count_orphaned_files(), 0)
            self.assertEqual(db.remove_orphaned_files(), 0)
            ids = _break(db)
            self.assertEqual(db.count_orphaned_files(), 3)
            before = _tables(db)

            db.begin_transaction()
            with db.conn:
                self.assertEqual(db.remove_orphaned_files(), 3)
            after = _tables(db)

            self.assertEqual(db.count_orphaned_files(), 0)
            for hash_id in ids["broken"]:
                for table in ("files", "shape_perceptual_hash_map", "shape_search_cache"):
                    self.assertIsNone(db.execute(f"SELECT 1 FROM {table} WHERE hash_id = ?;", (hash_id,)).fetchone())
            # Only the broken rows are gone: perceptual hashes (including the empty one), the search tree, the
            # version and every other file's rows and search cache are untouched.
            for table in ("shape_perceptual_hashes", "shape_vptree", "shape_maintenance_branch_regen", "version"):
                self.assertEqual(after[table], before[table], table)
            self.assertEqual(after["files"], _without(before["files"], ids["broken"], 0))
            self.assertEqual(
                after["shape_perceptual_hash_map"], _without(before["shape_perceptual_hash_map"], ids["broken"], 1)
            )
            self.assertEqual(after["shape_search_cache"], _without(before["shape_search_cache"], ids["broken"], 0))
            self.assertEqual(db.get_phash(ids["empty_phash_id"]), b"")
            db.close()

    def test_queued_file_keeps_its_files_row(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "orphans.sqlite", _make_hashes(seed=4))
            ids = _break(db)
            db.add_to_phashed_files_queue("a" * 64, b"\x01" * 32)
            db.commit()
            db.begin_transaction()
            with db.conn:
                db.remove_orphaned_files()
            self.assertEqual(db.get_hash_id("a" * 64), ids["broken"][0])
            self.assertEqual(db.count_orphaned_files(), 0)
            db.close()

    def test_messages(self):
        self.assertEqual(db_tools.orphans_check_message(407893, 0), "Checked 407893 videos: no broken entries.")
        self.assertEqual(
            db_tools.orphans_check_message(407984, 91),
            "WARNING: Checked 407984 videos: 91 videos are missing their perceptual hash. "
            "Run once with --prune-orphans to remove them. Your database is backed up automatically.",
        )
        self.assertEqual(
            db_tools.orphans_check_message(407984, 1),
            "WARNING: Checked 407984 videos: 1 video is missing its perceptual hash. "
            "Run once with --prune-orphans to remove it. Your database is backed up automatically.",
        )
        self.assertEqual(
            db_tools.orphans_check_message(2, 2, gui=True),
            "WARNING: Checked 2 videos: 2 videos are missing their perceptual hash. They are skipped, so nothing "
            "breaks. To remove them, run the command line version once with --prune-orphans. "
            "Your database is backed up automatically.",
        )
        self.assertEqual(
            db_tools.prune_orphans_message(91), "Removed 91 broken entries (they will be re-hashed on the next run)."
        )
        self.assertEqual(db_tools.prune_orphans_message(0), "No broken entries to remove.")

    def test_run_prune_orphans_is_all_or_nothing(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "orphans.sqlite", _make_hashes(seed=8))
            _break(db)
            before = _tables(db)
            real = DedupeDB.DedupeDb.remove_orphaned_files

            def remove_then_interrupt(self):
                real(self)
                raise KeyboardInterrupt

            with mock.patch.object(DedupeDB.DedupeDb, "remove_orphaned_files", remove_then_interrupt):
                self.assertEqual(db_tools.run_prune_orphans(db), 0)
            self.assertEqual(_tables(db), before)
            db.close()


class _FakeClient:
    def get_api_version(self):
        return 3

    def get_hydrus_api_version(self):
        return 80

    def verify_permissions(self):
        pass


class _FakeDeduplicator:
    def __init__(self, *args, **kwargs):
        pass

    def deduplicate(self, skip_hashing):
        return 0


class TestFlags(unittest.TestCase):
    """--prune-orphans and --relink-search-tree through the real CLI main(), without Hydrus (stubbed)."""

    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()
        self._old_db_dir = DedupeDB.get_db_dir()

    def tearDown(self):
        DedupeDB.set_db_dir(self._old_db_dir)
        logging.disable(logging.NOTSET)  # main() disables logging when not verbose

    def _make(self, tmpdir: str, old_backups: int = 0) -> tuple[Path, list[Path]]:
        folder = Path(tmpdir)
        db = _new_db(tmpdir, "videohashes.sqlite", _make_hashes(seed=9))
        _break(db)
        db.close()
        old = [folder / f"videohashes.backup-2020010{i}-000000.sqlite" for i in range(1, old_backups + 1)]
        for f in old:
            f.write_bytes(b"x")
        return folder, old

    def _run(self, folder: Path, **flags) -> dict:
        calls = {}
        with (
            mock.patch.object(entrypoint, "create_client", return_value=_FakeClient()),
            mock.patch.object(entrypoint, "HydrusVideoDeduplicator", _FakeDeduplicator),
            mock.patch.object(entrypoint, "check_database", wraps=db_tools.check_database) as w,
            mock.patch.object(entrypoint, "run_backup", wraps=db_tools.run_backup) as b,
            mock.patch.object(db_tools, "print_user", wraps=db_tools.print_user) as p,
        ):
            entrypoint.main(api_key="x" * 64, api_url="http://127.0.0.1:1", dedup_database_dir=folder, **flags)
        calls["check"], calls["backup"] = w, b
        calls["printed"] = [(c.args[0], c.kwargs.get("default_style")) for c in p.call_args_list]
        return calls

    @staticmethod
    def _open(folder: Path, name: str = "videohashes.sqlite") -> DedupeDB.DedupeDb:
        db = DedupeDB.DedupeDb(folder, name)
        db.init_connection()
        return db

    @staticmethod
    def _backups(folder: Path) -> list[str]:
        return sorted(f.name for f in folder.iterdir() if DedupeDB.backup_file_pattern().match(f.name))

    def test_prune_orphans_backs_up_first_and_removes_only_broken_rows(self):
        with TemporaryDirectory() as tmpdir:
            folder, _ = self._make(tmpdir)
            db = self._open(folder)
            before = _tables(db)
            broken = [h for (h,) in db.execute(db.ORPHANED_FILES_QUERY)]
            db.close()
            calls = self._run(folder, prune_orphans=True)
            self.assertFalse(calls["check"].call_args.kwargs["check_orphans"])  # its fix flag is given
            self.assertTrue(calls["check"].call_args.kwargs["check_tree"])
            calls["backup"].assert_called_once()
            self.assertEqual(calls["printed"], [])  # no old backups, so nothing deleted
            (backup_name,) = self._backups(folder)
            backup = self._open(folder, backup_name)
            self.assertEqual(_tables(backup), before)  # the backup is from before any change
            backup.close()
            db = self._open(folder)
            after = _tables(db)
            self.assertEqual(db.count_orphaned_files(), 0)
            db.close()
            for table in ("shape_perceptual_hashes", "shape_vptree", "shape_maintenance_branch_regen", "version"):
                self.assertEqual(after[table], before[table], table)
            self.assertEqual(after["shape_search_cache"], _without(before["shape_search_cache"], broken, 0))
            self.assertEqual(after["files"], _without(before["files"], broken, 0))

    def test_backup_rotation_is_the_same_for_every_flag(self):
        for flag in ("prune_orphans", "relink_search_tree", "backup"):
            with self.subTest(flag=flag), TemporaryDirectory() as tmpdir:
                folder, old = self._make(tmpdir, old_backups=3)
                calls = self._run(folder, **{flag: True})
                calls["backup"].assert_called_once()
                backups = self._backups(folder)
                self.assertEqual(len(backups), 3)
                self.assertEqual(backups[:2], [old[1].name, old[2].name])  # the oldest was deleted
                self.assertEqual(calls["printed"], [(db_tools.deleted_backup_message(old[0]), None)])

    def test_failed_delete_of_old_backup_is_a_yellow_warning(self):
        with TemporaryDirectory() as tmpdir:
            folder, old = self._make(tmpdir, old_backups=3)
            real_remove = os.remove

            def remove(path):
                if Path(path) == old[0]:
                    raise PermissionError(13, "Access is denied", str(path))
                real_remove(path)

            with mock.patch.object(DedupeDB.os, "remove", side_effect=remove):
                calls = self._run(folder, prune_orphans=True)
            warning = db_tools.delete_backup_failed_message(old[0], PermissionError(13, "Access is denied"))
            self.assertTrue(warning.endswith(": Access is denied"))
            self.assertEqual(calls["printed"], [(warning, "yellow")])
            self.assertTrue(old[0].exists())
            db = self._open(folder)
            self.assertEqual(db.count_orphaned_files(), 0)  # the run went on
            db.close()

    def test_relink_backs_up_and_still_skips_broken_entries(self):
        with TemporaryDirectory() as tmpdir:
            folder, _ = self._make(tmpdir)
            calls = self._run(folder, relink_search_tree=True)
            calls["backup"].assert_called_once()
            self.assertTrue(calls["check"].call_args.kwargs["check_orphans"])
            self.assertFalse(calls["check"].call_args.kwargs["check_tree"])  # its fix flag is given
            self.assertEqual(len(self._backups(folder)), 1)
            db = self._open(folder)
            self.assertEqual(db.count_orphaned_files(), 3)  # skipped, not deleted
            db.close()

    def test_prune_orphans_runs_before_relink(self):
        with TemporaryDirectory() as tmpdir:
            folder, _ = self._make(tmpdir)
            real_relink = DedupeDB.DedupeDb.relink_search_tree
            with mock.patch.object(
                DedupeDB.DedupeDb, "relink_search_tree", autospec=True, side_effect=real_relink
            ) as relink:
                calls = self._run(folder, prune_orphans=True, relink_search_tree=True)
            calls["backup"].assert_called_once()  # one backup, not one per flag
            relink.assert_called_once()
            db = self._open(folder)
            self.assertEqual(db.count_orphaned_files(), 0)
            phashes = db.execute("SELECT COUNT(*) FROM shape_perceptual_hashes;").fetchone()[0]
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shape_vptree;").fetchone()[0], phashes)
            db.close()

    def test_normal_run_only_warns(self):
        with TemporaryDirectory() as tmpdir:
            folder, _ = self._make(tmpdir)
            db = self._open(folder)
            before = _tables(db)
            db.close()
            calls = self._run(folder)
            self.assertEqual(calls["check"].call_args.kwargs, {"check_orphans": True, "check_tree": True})
            calls["backup"].assert_not_called()
            self.assertEqual(self._backups(folder), [])
            db = self._open(folder)
            self.assertEqual(_tables(db), before)
            db.close()


if __name__ == "__main__":
    unittest.main()
