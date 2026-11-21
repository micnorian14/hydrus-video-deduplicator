from __future__ import annotations

import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator import db_tools
from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes
from .test_relink import _new_db, _tables

THRESHOLDS = (1, 26, 51, 101)


def _searches(db: DedupeDB.DedupeDb, hash_ids) -> dict:
    tree = vptree.VpTreeManager(db)
    out = {t: {h: sorted(tree.search_file(h, max_hamming_distance=t)) for h in hash_ids} for t in THRESHOLDS}
    db.commit()  # the searches leave an open transaction (temp tables); the app commits before this too
    return out


def _assert_valid_tree(test: unittest.TestCase, db: DedupeDB.DedupeDb) -> set:
    """
    Structural checks (one root, every node reachable once, parent pointers, exact populations, every mapped
    phash in the tree). Returns the (ancestor, descendant, side) relations, for checking the vantage point property.
    """
    rows = {r[0]: r for r in db.execute("SELECT * FROM shape_vptree;").fetchall()}
    phashes = {p for (p,) in db.execute("SELECT phash_id FROM shape_perceptual_hashes;")}
    mapped = {p for (p,) in db.execute("SELECT phash_id FROM shape_perceptual_hash_map;")}
    test.assertTrue(mapped <= set(rows))
    if not rows:
        return set()
    roots = [p for p, r in rows.items() if r[1] is None]
    test.assertEqual(len(roots), 1)
    size, order, stack = {}, [], [roots[0]]
    while stack:
        n = stack.pop()
        test.assertNotIn(n, order)
        order.append(n)
        test.assertIn(n, phashes)
        _, parent, radius, inner, ip, outer, op = rows[n]
        if inner is None and outer is None:
            test.assertIsNone(radius)
        for child in (inner, outer):
            if child is not None:
                test.assertIn(child, rows)
                test.assertEqual(rows[child][1], n)
                stack.append(child)
    test.assertEqual(sorted(order), sorted(rows))
    for n in reversed(order):
        _, parent, radius, inner, ip, outer, op = rows[n]
        test.assertEqual(ip, size.get(inner, 0))
        test.assertEqual(op, size.get(outer, 0))
        size[n] = 1 + ip + op
    relations = set()
    for n in rows:
        child, parent = n, rows[n][1]
        while parent is not None:
            side = "inner" if rows[parent][3] == child else "outer"
            relations.add((parent, n, side, rows[parent][2]))
            child, parent = parent, rows[parent][1]
    return relations


def _assert_relations_kept(test: unittest.TestCase, before: set, after: set):
    """Every node is still on a side of each ancestor it was already on, or inside a radius-101 chain node."""
    old = {(a, d, side) for (a, d, side, _) in before}
    for a, d, side, radius in after:
        if (a, d, side) not in old:
            test.assertEqual((side, radius), ("inner", vptree.MAX_DISTANCE), (a, d))


class TestPruneDeletedFiles(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _check_prune(self, relink: bool, seed: int):
        vptree.TemporaryIntegerTableNameCache()  # fresh temp-table name cache for this DB connection
        hashes = _make_hashes(seed=seed)
        rng = random.Random(seed)
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "prune.sqlite", hashes)
            if relink:
                db.begin_transaction()
                with db.conn:
                    db.relink_search_tree()
            db.add_to_phashed_files_queue("ab" * 32, hashes[0])  # queued, deleted in Hydrus
            db.add_to_phashed_files_queue("cd" * 32, hashes[1])  # queued, still in Hydrus
            db.execute("UPDATE shape_search_cache SET searched_distance = 51 WHERE hash_id % 2 = 0;")
            db.commit()
            relations_before = _assert_valid_tree(self, db)
            files = dict(db.execute("SELECT hash_id, file_hash FROM files;").fetchall())
            gone = set(rng.sample(sorted(files), len(files) // 3))
            gone.add(db.get_hash_id(f"{len(hashes) - 2:064x}"))  # shares its phash with a kept file (hashes[3])
            gone.discard(db.get_hash_id(f"{3:064x}"))
            before = _searches(db, sorted(files))
            cache_before = dict(db.execute("SELECT hash_id, searched_distance FROM shape_search_cache;").fetchall())
            existing = [files[h].upper() for h in files if h not in gone] + [
                "cd" * 32,
                "ef" * 32,
            ]  # case must not matter

            db.begin_transaction()
            with db.conn:
                result = db.prune_files_not_in(existing)

            self.assertEqual(result.files, len(gone))
            self.assertEqual(result.queued, 1)
            self.assertEqual(result.videos, len(gone) + 1)
            remaining = dict(db.execute("SELECT hash_id, file_hash FROM files;").fetchall())
            self.assertEqual(set(remaining), set(files) - gone)
            queue = [f for (f,) in db.execute("SELECT file_hash FROM phashed_file_queue;")]
            self.assertEqual(queue, ["cd" * 32])
            cache_after = dict(db.execute("SELECT hash_id, searched_distance FROM shape_search_cache;").fetchall())
            self.assertEqual(cache_after, {h: d for h, d in cache_before.items() if h not in gone})
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM shape_perceptual_hash_map WHERE hash_id NOT IN (SELECT hash_id FROM files);"
                ).fetchone()[0],
                0,
            )
            used = {p for (p,) in db.execute("SELECT phash_id FROM shape_perceptual_hash_map;")}
            in_tree = {p for (p,) in db.execute("SELECT phash_id FROM shape_vptree;")}
            self.assertTrue(used <= in_tree)
            self.assertGreater(result.tree_nodes, 0)
            self.assertIsNone(
                db.execute(
                    "SELECT 1 FROM shape_maintenance_branch_regen "
                    "WHERE phash_id NOT IN (SELECT phash_id FROM shape_vptree);"
                ).fetchone()
            )
            _assert_relations_kept(self, relations_before, _assert_valid_tree(self, db))
            self.assertEqual(db.execute("PRAGMA integrity_check;").fetchone()[0], "ok")

            after = _searches(db, sorted(remaining))
            for t in THRESHOLDS:
                for h in remaining:
                    expected = [(o, d) for (o, d) in before[t][h] if o not in gone]
                    self.assertEqual(after[t][h], expected, f"threshold {t} hash_id {h}")

            # Pruning again finds nothing and changes nothing.
            tables = _tables(db)
            db.begin_transaction()
            with db.conn:
                again = db.prune_files_not_in(existing)
            self.assertEqual((again.videos, again.tree_nodes), (0, 0))
            self.assertEqual(_tables(db), tables)
            db.close()

    def test_prune_vp_tree(self):
        for seed in (11, 12, 13):
            self._check_prune(relink=False, seed=seed)

    def test_prune_chain_tree(self):
        self._check_prune(relink=True, seed=21)

    def test_splice_root_and_everything(self):
        hashes = _make_hashes(seed=5)
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "all.sqlite", hashes)
            relations_before = _assert_valid_tree(self, db)
            root = db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()[0]
            keep = [
                f
                for (h, f) in db.execute("SELECT hash_id, file_hash FROM files;").fetchall()
                if db.get_phash_id_from_hash_id(h) != root
            ]
            db.begin_transaction()
            with db.conn:
                db.prune_files_not_in(keep)
            _assert_relations_kept(self, relations_before, _assert_valid_tree(self, db))
            self.assertIsNone(db.execute("SELECT 1 FROM shape_vptree WHERE phash_id = ?;", (root,)).fetchone())
            db.begin_transaction()
            with db.conn:
                result = db.prune_files_not_in(["00" * 32])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shape_vptree;").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM files;").fetchone()[0], 0)
            self.assertEqual(result.files, len(keep))
            db.close()

    def test_run_prune_skips_when_hydrus_fails_or_is_empty(self):
        class FakeClient:
            def __init__(self, result):
                self.result = result

            def get_all_video_hashes_in_hydrus(self):
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result

        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "skip.sqlite", _make_hashes(seed=3))
            tables = _tables(db)
            self.assertIsNone(db_tools.run_prune(db, FakeClient([])))
            self.assertIsNone(db_tools.run_prune(db, FakeClient(RuntimeError("no connection " + "a" * 64))))
            self.assertEqual(_tables(db), tables)
            db.close()

    def test_prune_rolls_back_on_interrupt(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "rollback.sqlite", _make_hashes(seed=4))
            tables = _tables(db)
            original = vptree.splice_out_tree_nodes

            def boom(*args):
                raise KeyboardInterrupt

            DedupeDB.splice_out_tree_nodes = boom
            try:

                class Client:
                    def get_all_video_hashes_in_hydrus(self):
                        return ["00" * 32]

                self.assertIsNone(db_tools.run_prune(db, Client()))
            finally:
                DedupeDB.splice_out_tree_nodes = original
            self.assertEqual(_tables(db), tables)
            db.close()

    def test_messages(self):
        r = DedupeDB.PruneResult(videos=1, files=1, queued=0, tree_nodes=1, perceptual_hashes=1)
        self.assertEqual(
            db_tools.prune_message(r), "Pruned 1 deleted video from the database and 1 entry from the search tree."
        )
        r = DedupeDB.PruneResult(videos=63019, files=63019, queued=0, tree_nodes=2, perceptual_hashes=2)
        self.assertEqual(
            db_tools.prune_message(r),
            "Pruned 63019 deleted videos from the database and 2 entries from the search tree.",
        )
        r = DedupeDB.PruneResult(videos=0, files=0, queued=0, tree_nodes=0, perceptual_hashes=0)
        self.assertEqual(db_tools.prune_message(r), "No deleted videos to prune.")


class TestBackup(unittest.TestCase):
    def test_backup_copies_and_keeps_three(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "videohashes.sqlite", _make_hashes(seed=2))
            tables = _tables(db)
            db.close()
            folder = Path(tmpdir)
            old = [folder / f"videohashes.backup-2020010{i}-000000.sqlite" for i in range(1, 5)]
            others = [
                folder / "videohashes.backup-notadate.sqlite",
                folder / "other.backup-20200101-000000.sqlite",
                folder / "videohashes.backup-20200101-000000.sqlite.partial",
            ]
            for f in old + others:
                f.write_bytes(b"x")
            path = DedupeDB.backup_db(folder / "videohashes.sqlite")
            self.assertTrue(DedupeDB.backup_file_pattern().match(path.name))
            backups = sorted(f.name for f in folder.iterdir() if DedupeDB.backup_file_pattern().match(f.name))
            self.assertEqual(backups, [old[2].name, old[3].name, path.name])
            for f in others:
                self.assertTrue(f.exists())
            copy = DedupeDB.DedupeDb(folder, path.name)
            copy.init_connection()
            self.assertEqual(_tables(copy), tables)
            copy.close()
            self.assertIsNone(DedupeDB.backup_db(folder / "missing.sqlite"))


if __name__ == "__main__":
    unittest.main()
