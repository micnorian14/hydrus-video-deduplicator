from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hydrusvideodeduplicator.db import DedupeDB, vptree
from hydrusvideodeduplicator.db.fast_search import MAX_DISTANCE, SearchDistanceIndex, fast_search_enabled

from .test_fast_search import _blob, _flip, _frame, _make_hashes


def _extra_hashes(base: list[bytes], seed: int) -> list[bytes]:
    """New files for an existing tree: near-duplicates that branch inside nonzero radii, exact duplicates,
    trimmed and repeated copies, unrelated videos, an empty hash and a low-entropy (all black) video."""
    rng = random.Random(seed)
    usable = [h for h in base if h]
    out = []
    for k in range(60):
        frames = [usable[k % len(usable)][i : i + 32] for i in range(0, len(usable[k % len(usable)]), 32)]
        kind = k % 6
        if kind == 0:
            frames = [
                _flip(rng, int.from_bytes(f, "little"), rng.randint(10, 33)).to_bytes(32, "little") for f in frames
            ]
        elif kind == 1 and len(frames) > 1:
            frames = frames[1:]
        elif kind == 2:
            frames = frames + frames[:2]
        elif kind == 3:
            frames = [_frame(rng).to_bytes(32, "little") for _ in range(rng.randint(1, 12))]
        elif kind == 4:
            out.append(usable[k % len(usable)])  # exact duplicate of a file already in the tree
            continue
        out.append(b"".join(frames))
    out.insert(7, b"")  # empty hash: never indexed, always calculate_distance
    out.insert(20, bytes(32 * 9))  # nine all-zero frames
    out.insert(21, bytes(32 * 9))  # and the same hash again
    out.insert(30, out[3])  # a duplicate of a file added in this same batch
    return out


def _add_files(db: DedupeDB.DedupeDb, hashes: list[bytes], start: int, commit_every: int, snapshots: list):
    """Add files the way the tree building phase does: in a transaction, committing every few files."""
    with vptree.fast_tree_building(db):
        db.begin_transaction()
        for n, (i, phash) in enumerate(enumerate(hashes, start), 1):
            file_hash = f"{i:064x}"
            db.add_file(file_hash)
            db.add_perceptual_hash(phash)
            db.associate_file_with_perceptual_hash(file_hash, phash)
            if n % commit_every == 0:
                db.commit()
                snapshots.append(_tree(db))
                db.begin_transaction()
    db.commit()
    snapshots.append(_tree(db))


def _tree(db: DedupeDB.DedupeDb) -> dict:
    return {
        "vptree": db.execute(
            "SELECT phash_id, parent_id, radius, typeof(radius), inner_id, inner_population, outer_id, outer_population"
            " FROM shape_vptree ORDER BY phash_id;"
        ).fetchall(),
        "regen": db.execute("SELECT * FROM shape_maintenance_branch_regen ORDER BY 1;").fetchall(),
        "map": db.execute("SELECT * FROM shape_perceptual_hash_map ORDER BY 1, 2;").fetchall(),
    }


def _new(tmpdir: str, name: str, base: list[bytes], relink: bool) -> DedupeDB.DedupeDb:
    db = DedupeDB.DedupeDb(Path(tmpdir), name)
    db.init_connection()
    db.create_tables()
    for i, phash in enumerate(base):
        file_hash = f"{i:064x}"
        db.add_file(file_hash)
        db.add_perceptual_hash(phash)
        db.associate_file_with_perceptual_hash(file_hash, phash)
    db.commit()
    if relink:
        db.begin_transaction()
        with db.conn:
            db.relink_search_tree()
    return db


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestSparseDistances(unittest.TestCase):
    def test_sparse_distances_match_calculate_distance(self):
        hashes = [h for h in _make_hashes(seed=5) if h] + [bytes(32 * 5), bytes(32 * 3) + bytes([1]) * 32]
        index = SearchDistanceIndex(list(range(len(hashes))), hashes)
        for q in hashes + [_blob([_frame(random.Random(3))] * 4)]:
            positions, distances = index.sparse_distances_for(q)
            expected = [vptree.calculate_distance(q, h) for h in hashes]
            self.assertEqual(positions.tolist(), sorted(set(positions.tolist())))
            full = [MAX_DISTANCE] * len(hashes)
            for p, d in zip(positions.tolist(), distances.tolist()):
                self.assertLess(d, MAX_DISTANCE)  # listed nodes have at least one matching frame
                full[p] = d
            self.assertEqual(full, expected)
            self.assertEqual(index.distances_for(q), expected)
        self.assertIsNone(index.sparse_distances_for(b""))
        self.assertIsNone(index.sparse_distances_for(b"\x00" * 33))
        self.assertEqual(vptree.MAX_DISTANCE, MAX_DISTANCE)


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestFastInsertPrefix(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _both(self, base, extra, relink, commit_every):
        results = {}
        with TemporaryDirectory() as tmpdir:
            for mode in ("0", "1"):
                with mock.patch.dict(os.environ, {"HVD_FAST_INSERT": mode}):
                    vptree.TemporaryIntegerTableNameCache()
                    db = _new(tmpdir, f"t{mode}.sqlite", base, relink)
                    snapshots = []
                    _add_files(db, extra, len(base), commit_every, snapshots)
                    results[mode] = snapshots
                    db.close()
        return results

    def test_inserts_into_a_chain_are_identical(self):
        base = _make_hashes(seed=11)
        extra = _extra_hashes(base, seed=12)
        for commit_every in (1, 5, 1000):
            results = self._both(base, extra, relink=True, commit_every=commit_every)
            self.assertEqual(results["0"], results["1"])  # every committed state, every row and radius
        radii = {r[2] for r in results["1"][-1]["vptree"]}
        self.assertIn(MAX_DISTANCE, radii)  # the chain (prefix) part
        self.assertTrue(any(r is not None and r < MAX_DISTANCE for r in radii))  # branches inside smaller radii
        self.assertTrue(any(r[6] is not None for r in results["1"][-1]["vptree"]))  # and outer children

    def test_inserts_into_a_general_tree_are_identical(self):
        base = [h for h in _make_hashes(seed=21)][:30]
        extra = _extra_hashes(_make_hashes(seed=21), seed=22)
        results = self._both(base, extra, relink=False, commit_every=4)
        self.assertEqual(results["0"], results["1"])

    def test_inserts_from_an_empty_tree_are_identical(self):
        extra = _extra_hashes(_make_hashes(seed=31), seed=32)
        results = self._both([], extra, relink=False, commit_every=3)
        self.assertEqual(results["0"], results["1"])

    def test_prefix_is_used_and_counts_are_written_before_each_commit(self):
        base = _make_hashes(seed=41)
        extra = _extra_hashes(base, seed=42)[:12]
        with TemporaryDirectory() as tmpdir:
            db = _new(tmpdir, "p.sqlite", base, relink=True)
            with vptree.fast_tree_building(db) as fast:
                self.assertTrue(fast)
                state = vptree._active_fast_insert
                db.begin_transaction()
                for i, phash in enumerate(extra, len(base)):
                    db.add_file(f"{i:064x}")
                    db.add_perceptual_hash(phash)
                    db.associate_file_with_perceptual_hash(f"{i:064x}", phash)
                self.assertGreater(len(state.prefix_ids), len(base) // 2)
                self.assertTrue(state.pending)
                db.commit()  # DedupeDb.commit flushes first
                self.assertEqual(state.pending, [])
            db.close()

    def test_rolled_back_file_is_not_counted(self):
        base = _make_hashes(seed=51)
        extra = _extra_hashes(base, seed=52)[:10]
        with TemporaryDirectory() as tmpdir:
            trees = {}
            for mode in ("0", "1"):
                with mock.patch.dict(os.environ, {"HVD_FAST_INSERT": mode}):
                    vptree.TemporaryIntegerTableNameCache()
                    db = _new(tmpdir, f"r{mode}.sqlite", base, relink=True)
                    with vptree.fast_tree_building(db):
                        db.begin_transaction()
                        for i, phash in enumerate(extra, len(base)):
                            if i == len(base) + 6:  # this file's queue item fails after its tree insert
                                db.execute("SAVEPOINT item;")
                                db.add_perceptual_hash(phash)
                                tree = vptree.VpTreeManager(db)
                                tree.add_leaf(db.get_phash_id(phash), phash)
                                db.execute("ROLLBACK TO item;")
                                db.execute("RELEASE item;")
                                break
                            db.add_file(f"{i:064x}")
                            db.add_perceptual_hash(phash)
                            db.associate_file_with_perceptual_hash(f"{i:064x}", phash)
                    db.commit()
                    trees[mode] = _tree(db)
                    db.close()
            self.assertEqual(trees["0"], trees["1"])

    def test_kill_switches(self):
        with TemporaryDirectory() as tmpdir:
            db = _new(tmpdir, "k.sqlite", _make_hashes(seed=61), relink=True)
            with mock.patch.dict(os.environ, {"HVD_FAST_INSERT": "0"}):
                with vptree.fast_tree_building(db) as fast:
                    self.assertFalse(fast)
                    self.assertIsNone(vptree._active_fast_insert)
            with mock.patch.dict(os.environ, {"HVD_FAST_SEARCH": "0"}):
                self.assertFalse(vptree.VpTreeManager(db).enable_fast_search_index())
            with mock.patch.dict(os.environ, {"HVD_SCAN_SEARCH": "0"}):
                tree = vptree.VpTreeManager(db)
                self.assertTrue(tree.enable_fast_search_index())
                self.assertFalse(tree.enable_fast_search_scan())
            db.close()


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestFastSearchWalk(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def test_search_identical_on_chain_with_branching_tail(self):
        base = _make_hashes(seed=71)
        extra = _extra_hashes(base, seed=72)
        with TemporaryDirectory() as tmpdir:
            db = _new(tmpdir, "s.sqlite", base, relink=True)
            _add_files(db, extra, len(base), 1000, [])
            hash_ids = [r[0] for r in db.execute("SELECT hash_id FROM shape_perceptual_hash_map ORDER BY hash_id;")]
            for radius in (1, 16, 26, 51, 76, 100, 101):
                expected = [sorted(vptree.VpTreeManager(db).search_file(h, radius)) for h in hash_ids]
                index_only = vptree.VpTreeManager(db)
                self.assertTrue(index_only.enable_fast_search_index())
                scan = vptree.VpTreeManager(db)
                self.assertTrue(scan.enable_fast_search_index())
                self.assertTrue(scan.enable_fast_search_scan())
                for h, want in zip(hash_ids, expected):
                    self.assertEqual(sorted(index_only.search_file(h, radius)), want)
                    self.assertEqual(sorted(scan.search_file(h, radius)), want)
                if radius == 51:
                    segments = scan._search_scan_segments(radius)[0]
                    self.assertTrue(any(len(ids) > 1 for ids, _next, _unindexed in segments))  # straight runs used
            db.close()
