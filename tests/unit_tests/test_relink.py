from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes


def _new_db(tmpdir: str, name: str, hashes: list[bytes], start: int = 0) -> DedupeDB.DedupeDb:
    db = DedupeDB.DedupeDb(Path(tmpdir), name)
    db.init_connection()
    db.create_tables()
    _add(db, hashes, start)
    return db


def _add(db: DedupeDB.DedupeDb, hashes: list[bytes], start: int = 0):
    for i, phash in enumerate(hashes, start):
        file_hash = f"{i:064x}"
        db.add_file(file_hash)
        db.add_perceptual_hash(phash)
        db.associate_file_with_perceptual_hash(file_hash, phash)
    db.commit()


def _tables(db: DedupeDB.DedupeDb) -> dict:
    names = [n for (n,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;").fetchall()]
    return {n: sorted(db.execute(f"SELECT * FROM {n};").fetchall(), key=repr) for n in names}


class TestRelinkSearchTree(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()  # fresh temp-table name cache for this test's own DB connection

    def test_relink_keeps_search_results_and_makes_a_valid_chain(self):
        hashes = _make_hashes(seed=7)
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "relink.sqlite", hashes)
            # A file whose perceptual hash row is missing must be skipped, not deleted.
            db.execute("INSERT INTO files ( file_hash ) VALUES ( 'broken' );")
            broken_id = db.get_hash_id("broken")
            db.execute("INSERT INTO shape_perceptual_hash_map ( phash_id, hash_id ) VALUES ( 999999, ? );", (broken_id,))
            db.commit()
            hash_ids = [r[0] for r in db.execute("SELECT hash_id FROM shape_search_cache ORDER BY hash_id;")]
            before = {
                t: {h: sorted(vptree.VpTreeManager(db).search_file(h, max_hamming_distance=t)) for h in hash_ids}
                for t in (1, 26, 51, 101)
            }

            db.commit()  # the searches above leave an open transaction (temp tables); the app commits before this too
            db.begin_transaction()
            with db.conn:
                stats = db.relink_search_tree()

            phash_count = db.execute("SELECT COUNT(*) FROM shape_perceptual_hashes;").fetchone()[0]
            self.assertEqual(stats["tree_nodes"], phash_count)
            self.assertEqual(stats["skipped_missing_phash"], 1)
            self.assertEqual(stats["files_to_search"], len(hash_ids))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM shape_perceptual_hash_map;").fetchone()[0], len(hash_ids) + 1)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM shape_search_cache WHERE searched_distance IS NULL;").fetchone()[0],
                len(hash_ids),
            )
            self.assertIsNone(db.execute("SELECT 1 FROM shape_maintenance_branch_regen;").fetchone())

            # one chain from the root that reaches every node exactly once
            rows = {r[0]: r for r in db.execute("SELECT * FROM shape_vptree;").fetchall()}
            node = db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchall()
            self.assertEqual(len(node), 1)
            node, seen = node[0][0], []
            while node is not None and len(seen) <= len(rows):
                seen.append(node)
                node = db.execute("SELECT inner_id FROM shape_vptree WHERE phash_id = ?;", (node,)).fetchone()[0]
            self.assertEqual(sorted(seen), sorted(rows))

            after = {
                t: {h: sorted(vptree.VpTreeManager(db).search_file(h, max_hamming_distance=t)) for h in hash_ids}
                for t in (1, 26, 51, 101)
            }
            for t in before:  # the old tree here is intact, so results must be identical
                self.assertEqual(after[t], before[t], f"threshold {t}")
            db.close()

    def test_inserts_after_relink_identical_fast_and_original(self):
        hashes = _make_hashes(seed=8)
        old, new = hashes[: len(hashes) // 2], hashes[len(hashes) // 2 :]
        results = []
        with TemporaryDirectory() as tmpdir:
            for fast in (True, False):
                db = _new_db(tmpdir, f"insert_{fast}.sqlite", old)
                db.begin_transaction()
                with db.conn:
                    db.relink_search_tree()
                prev = os.environ.get("HVD_FAST_INSERT")
                os.environ["HVD_FAST_INSERT"] = "1" if fast else "0"
                try:
                    with vptree.fast_tree_building(db) as active:
                        self.assertEqual(active, fast)
                        _add(db, new, start=1000)
                finally:
                    if prev is None:
                        del os.environ["HVD_FAST_INSERT"]
                    else:
                        os.environ["HVD_FAST_INSERT"] = prev
                results.append(_tables(db))
                db.close()
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
