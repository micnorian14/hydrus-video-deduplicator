from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator.db import DedupeDB, vptree
from hydrusvideodeduplicator.db.fast_search import fast_search_enabled

from .test_fast_search import _make_hashes

THRESHOLDS = (1, 26, 50, 51, 52, 76, 100, 101)


def _add(db: DedupeDB.DedupeDb, hashes: list[bytes], start: int = 0):
    for i, phash in enumerate(hashes, start):
        file_hash = f"{i:064x}"
        db.add_file(file_hash)
        db.add_perceptual_hash(phash)
        db.associate_file_with_perceptual_hash(file_hash, phash)
    db.commit()


def _search_all(db: DedupeDB.DedupeDb, mode: str, threshold: int) -> list:
    tree = vptree.VpTreeManager(db)
    if mode in ("index", "scan"):
        assert tree.enable_fast_search_index()
    if mode == "scan":
        assert tree.enable_fast_search_scan()
    hash_ids = [r[0] for r in db.execute("SELECT hash_id FROM shape_search_cache ORDER BY hash_id;").fetchall()]
    return [tree.search_file(h, max_hamming_distance=threshold) for h in hash_ids]


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestFastSearchScan(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()  # fresh temp-table name cache for this test's own DB connection

    def _assert_identical(self, db: DedupeDB.DedupeDb):
        for threshold in THRESHOLDS:
            original = _search_all(db, "original", threshold)
            self.assertEqual(_search_all(db, "index", threshold), original, f"index, threshold {threshold}")
            self.assertEqual(_search_all(db, "scan", threshold), original, f"scan, threshold {threshold}")

    def test_identical_on_a_normal_tree(self):
        """A tree built by add_leaf has all kinds of radii: the scan must give exactly the original results."""
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "normal.sqlite")
            db.init_connection()
            db.create_tables()
            _add(db, _make_hashes(seed=5))
            self._assert_identical(db)
            db.close()

    def test_identical_on_a_relinked_chain_with_later_inserts(self):
        """Like a real library after --relink-search-tree plus newly hashed files added by add_leaf afterwards."""
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "chain.sqlite")
            db.init_connection()
            db.create_tables()
            hashes = _make_hashes(seed=21)  # includes an empty hash: in the chain, but not in the fast index
            _add(db, hashes)
            db.begin_transaction()
            with db.conn:
                db.relink_search_tree()
            _add(db, _make_hashes(seed=22)[:40] + [b""], start=10_000)  # later inserts: other radii, outer children
            radii = {r for (r,) in db.execute("SELECT radius FROM shape_vptree;").fetchall()}
            self.assertIn(vptree.MAX_DISTANCE, radii)
            self.assertGreater(len(radii - {vptree.MAX_DISTANCE, None}), 0)

            tree = vptree.VpTreeManager(db)
            tree.enable_fast_search_index()
            self.assertTrue(tree.enable_fast_search_scan())
            hash_id = db.execute("SELECT hash_id FROM shape_search_cache LIMIT 1;").fetchone()[0]
            tree.search_file(hash_id, max_hamming_distance=51)
            prefix = tree._search_scan_prefix(51)
            self.assertIsNotNone(prefix)
            self.assertGreater(len(prefix[1]), 100)  # the scan really covers most of the chain here
            self.assertEqual(len(prefix[3]), 1)  # ...including the unindexed empty-hash node

            self._assert_identical(db)

            # If an unindexed node's distance ever let the walk stop there, the scan must give it back to the walk.
            original_calculate_distance = vptree.calculate_distance

            def odd_distance(a, b):
                return vptree.MAX_DISTANCE + 60 if b == b"" else original_calculate_distance(a, b)

            vptree.calculate_distance = odd_distance
            try:
                self._assert_identical(db)
            finally:
                vptree.calculate_distance = original_calculate_distance
            db.close()

    def test_prefix_stops_at_a_loop(self):
        """A damaged tree with a loop must not hang the prefix computation (the seen set bounds it)."""
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "loop.sqlite")
            db.init_connection()
            db.create_tables()
            _add(db, [h for h in _make_hashes(seed=3) if h][:30])
            db.begin_transaction()
            with db.conn:
                db.relink_search_tree()
            root = db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()[0]
            last = db.execute(
                "SELECT phash_id FROM shape_vptree WHERE inner_id IS NULL AND outer_id IS NULL;"
            ).fetchone()[0]
            db.execute(
                "UPDATE shape_vptree SET radius = ?, inner_id = ? WHERE phash_id = ?;",
                (vptree.MAX_DISTANCE, root, last),
            )
            db.commit()
            tree = vptree.VpTreeManager(db)
            tree.enable_fast_search_index()
            self.assertTrue(tree.enable_fast_search_scan())
            tree._root_node_perceptual_hash_id = root
            positions, ids, next_id, unindexed = tree._search_scan_prefix(51)
            self.assertEqual(len(ids), 30)
            self.assertEqual(next_id, root)  # handed back to the original walk, which behaves exactly as before
            db.close()

    def test_can_be_disabled(self):
        old = os.environ.get("HVD_SCAN_SEARCH")
        os.environ["HVD_SCAN_SEARCH"] = "0"
        try:
            self.assertFalse(vptree.scan_search_enabled())
            with TemporaryDirectory() as tmpdir:
                db = DedupeDB.DedupeDb(Path(tmpdir), "off.sqlite")
                db.init_connection()
                db.create_tables()
                _add(db, [h for h in _make_hashes(seed=4) if h][:10])
                tree = vptree.VpTreeManager(db)
                self.assertTrue(tree.enable_fast_search_index())
                self.assertFalse(tree.enable_fast_search_scan())
                db.close()
        finally:
            if old is None:
                del os.environ["HVD_SCAN_SEARCH"]
            else:
                os.environ["HVD_SCAN_SEARCH"] = old

    def test_needs_the_index(self):
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "noindex.sqlite")
            db.init_connection()
            db.create_tables()
            _add(db, [h for h in _make_hashes(seed=4) if h][:10])
            self.assertFalse(vptree.VpTreeManager(db).enable_fast_search_scan())
            db.close()


if __name__ == "__main__":
    unittest.main()
