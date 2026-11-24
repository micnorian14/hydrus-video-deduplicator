from __future__ import annotations

import os
import random
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator.db import DedupeDB, vptree
from hydrusvideodeduplicator.db import fast_search as fast_search_mod
from unittest import mock

from hydrusvideodeduplicator.db.fast_search import (
    FRAME_BYTES,
    HVD_FAST_SEARCH_MAX_GIB,
    SearchDistanceIndex,
    _BYTES_PER_FRAME,
    _OFFSETS_BYTES,
    _GIB,
    _reachable_phash_ids,
    estimated_peak_bytes,
    fast_search_build_error_warning,
    fast_search_cap_warning,
    fast_search_enabled,
    max_index_bytes,
    reset_fast_search_fallback_warning,
)


def _frame(rng: random.Random) -> int:
    return rng.getrandbits(256)


def _flip(rng: random.Random, x: int, k: int) -> int:
    for b in rng.sample(range(256), k):
        x ^= 1 << b
    return x


def _blob(frames: list[int]) -> bytes:
    return b"".join(f.to_bytes(32, "little") for f in frames)


def _make_hashes(seed: int = 1234) -> list[bytes]:
    """Synthetic vPDQ hashes: clusters of near-duplicates whose frames differ by 25..35 bits (around the 31 cutoff)."""
    rng = random.Random(seed)
    hashes = []
    for _ in range(25):
        base = [_frame(rng) for _ in range(rng.randint(1, 40))]
        hashes.append(_blob(base))
        for _ in range(rng.randint(1, 6)):
            variant = [_flip(rng, f, rng.randint(25, 35)) if rng.random() < 0.8 else _frame(rng) for f in base]
            variant = variant[rng.randint(0, len(variant) - 1) :]  # clips of different lengths
            if rng.random() < 0.3:
                variant = variant + variant[:3]  # repeated frames
            hashes.append(_blob(variant))
    hashes.append(hashes[3])  # identical phash shared by two files
    hashes.append(b"")  # empty hash (no usable frames) - must fall back to the original code
    return hashes


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestFastSearch(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()  # fresh temp-table name cache for this test's own DB connection
        reset_fast_search_fallback_warning()

    def test_distances_match_calculate_distance(self):
        hashes = [h for h in _make_hashes() if h]
        index = SearchDistanceIndex(list(range(len(hashes))), hashes)
        for q in hashes:
            fast = index.distances_for(q)
            self.assertEqual(fast, [vptree.calculate_distance(q, h) for h in hashes])
        self.assertIsNone(index.distances_for(b""))
        self.assertIsNone(index.distances_for(b"\x00" * 33))

    def test_search_file_identical_with_and_without_index(self):
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "videohashes.sqlite")
            db.init_connection()
            db.create_tables()
            for i, phash in enumerate(_make_hashes()):
                file_hash = f"{i:064x}"
                db.add_file(file_hash)
                db.add_perceptual_hash(phash)
                db.associate_file_with_perceptual_hash(file_hash, phash)
            db.commit()
            hash_ids = [r[0] for r in db.execute("SELECT hash_id FROM shape_search_cache ORDER BY hash_id;")]
            self.assertGreater(len(hash_ids), 50)

            for threshold in (1, 26, 51, 76, 101):
                original = vptree.VpTreeManager(db)
                expected = [original.search_file(h, max_hamming_distance=threshold) for h in hash_ids]

                fast = vptree.VpTreeManager(db)
                self.assertTrue(fast.enable_fast_search_index())
                got = [fast.search_file(h, max_hamming_distance=threshold) for h in hash_ids]
                self.assertEqual(got, expected, f"threshold {threshold}")  # same pairs, distances and order
            db.close()

    def test_fast_tree_building_identical(self):
        """Inserting files with vptree.fast_tree_building active must give exactly the same rows in every table."""

        def build(tmpdir: str, fast: bool):
            db = DedupeDB.DedupeDb(Path(tmpdir), "fast.sqlite" if fast else "orig.sqlite")
            db.init_connection()
            db.create_tables()
            db.commit()
            hashes = _make_hashes(seed=99)
            with vptree.fast_tree_building(db) if fast else nullcontext(False) as active:
                self.assertEqual(active, fast)
                for i, phash in enumerate(hashes + hashes[:10]):  # re-adding some exercises the already-in-tree path
                    file_hash = f"{i:064x}"
                    db.add_file(file_hash)
                    db.add_perceptual_hash(phash)
                    db.associate_file_with_perceptual_hash(file_hash, phash)
                    if i % 16 == 0:
                        db.commit()
            db.commit()
            tables = {}
            for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;").fetchall():
                tables[name] = sorted(db.execute(f"SELECT * FROM {name};").fetchall(), key=repr)
            db.close()
            return tables

        with TemporaryDirectory() as tmpdir:
            original = build(tmpdir, fast=False)
            fast = build(tmpdir, fast=True)
            self.assertGreater(len(original["shape_vptree"]), 50)
            self.assertEqual(fast, original)

    def test_can_be_disabled(self):
        old = os.environ.get("HVD_FAST_SEARCH")
        old_insert = os.environ.get("HVD_FAST_INSERT")
        os.environ["HVD_FAST_SEARCH"] = "0"
        os.environ["HVD_FAST_INSERT"] = "0"
        try:
            self.assertFalse(fast_search_enabled())
            self.assertFalse(vptree.fast_insert_enabled())
        finally:
            if old_insert is None:
                del os.environ["HVD_FAST_INSERT"]
            else:
                os.environ["HVD_FAST_INSERT"] = old_insert
            if old is None:
                del os.environ["HVD_FAST_SEARCH"]
            else:
                os.environ["HVD_FAST_SEARCH"] = old

    def _db_with_hashes(self, tmpdir: str, hashes: list[bytes]) -> DedupeDB.DedupeDb:
        db = DedupeDB.DedupeDb(Path(tmpdir), "videohashes.sqlite")
        db.init_connection()
        db.create_tables()
        for i, phash in enumerate(hashes):
            file_hash = f"{i:064x}"
            db.add_file(file_hash)
            db.add_perceptual_hash(phash)
            db.associate_file_with_perceptual_hash(file_hash, phash)
        db.commit()
        return db

    def _assert_indexes_identical(self, left: SearchDistanceIndex, right: SearchDistanceIndex):
        import numpy as np

        self.assertEqual(left.num_nodes, right.num_nodes)
        self.assertEqual(left.num_frames, right.num_frames)
        self.assertEqual(left.position_of, right.position_of)
        np.testing.assert_array_equal(left._words, right._words)
        np.testing.assert_array_equal(left._perm, right._perm)
        np.testing.assert_array_equal(left._offsets, right._offsets)
        np.testing.assert_array_equal(left._frame_owner, right._frame_owner)

    def test_init_skips_empty_null_and_non_multiple_blobs(self):
        import numpy as np

        valid_a = _blob([_frame(random.Random(1))])
        valid_b = _blob([_frame(random.Random(2)), _frame(random.Random(3))])
        index = SearchDistanceIndex(
            [10, 11, 12, 13, 14, 15],
            [valid_a, b"", b"\x00" * 33, None, valid_b, b"\x00" * 31],  # type: ignore[list-item]
        )
        self.assertEqual(index.num_nodes, 2)
        self.assertEqual(index.num_frames, 3)
        self.assertEqual(index.position_of, {10: 0, 14: 1})
        expected = SearchDistanceIndex([10, 14], [valid_a, valid_b])
        self._assert_indexes_identical(index, expected)
        got = index.sparse_distances_for(valid_a)
        exp = expected.sparse_distances_for(valid_a)
        np.testing.assert_array_equal(got[0], exp[0])
        np.testing.assert_array_equal(got[1], exp[1])

    def test_build_matches_direct_init_and_skips_invalid_hashes(self):
        import numpy as np

        hashes = [h for h in _make_hashes(seed=7) if h]
        with TemporaryDirectory() as tmpdir:
            db = self._db_with_hashes(tmpdir, hashes)
            rows = db.execute("SELECT phash_id FROM shape_perceptual_hashes;").fetchall()
            empty_id, bad_id, null_id = rows[0][0], rows[1][0], rows[2][0]
            db.execute(
                "UPDATE shape_perceptual_hashes SET phash = :phash WHERE phash_id = :phash_id",
                {"phash": b"", "phash_id": empty_id},
            )
            db.execute(
                "UPDATE shape_perceptual_hashes SET phash = :phash WHERE phash_id = :phash_id",
                {"phash": b"\x00" * 33, "phash_id": bad_id},
            )
            db.execute(
                "UPDATE shape_perceptual_hashes SET phash = :phash WHERE phash_id = :phash_id",
                {"phash": None, "phash_id": null_id},
            )
            db.commit()

            built = SearchDistanceIndex.build(db)
            self.assertIsNotNone(built)

            reachable = _reachable_phash_ids(db)
            ids: list[int] = []
            blobs: list[bytes] = []
            for phash_id, typ, length, phash in db.execute(
                "SELECT phash_id, typeof(phash), length(phash), phash FROM shape_perceptual_hashes;"
            ):
                if phash_id not in reachable:
                    continue
                if typ != "blob" or length is None or length <= 0 or length % FRAME_BYTES:
                    continue
                ids.append(int(phash_id))
                blobs.append(phash)

            # The two updated hashes and the NULL row must have been skipped.
            self.assertNotIn(empty_id, built.position_of)
            self.assertNotIn(bad_id, built.position_of)
            self.assertNotIn(null_id, built.position_of)
            self.assertEqual(len(ids), built.num_nodes)
            self.assertGreater(built.num_nodes, 10)

            direct = SearchDistanceIndex(ids, blobs)
            self._assert_indexes_identical(built, direct)
            mixed = SearchDistanceIndex([0, *ids, 999], [b"", *blobs, b"\x01" * 33])
            self._assert_indexes_identical(mixed, direct)
            for query in blobs[:8]:
                got = built.sparse_distances_for(query)
                exp = direct.sparse_distances_for(query)
                np.testing.assert_array_equal(got[0], exp[0])
                np.testing.assert_array_equal(got[1], exp[1])
            db.close()

    def test_offsets_and_perm_match_stable_searchsorted(self):
        import numpy as np

        rng = random.Random(0)
        blobs = [_blob([_frame(rng) for _ in range(20)]) for _ in range(15)]
        index = SearchDistanceIndex(list(range(len(blobs))), blobs)
        all_values = np.arange(65537)
        for w in range(4):
            block_group = index._words[w].view(np.uint16).reshape(index.num_frames, 4)
            for sub in range(4):
                j = 4 * w + sub
                column = block_group[:, sub]
                order = np.argsort(column, kind="stable")
                expected_offsets = np.searchsorted(column[order], all_values)
                np.testing.assert_array_equal(index._perm[j], order)
                np.testing.assert_array_equal(index._offsets[j], expected_offsets)

    def test_peak_memory_guard_uses_construction_peak(self):
        hashes = [h for h in _make_hashes() if h]
        with TemporaryDirectory() as tmpdir:
            db = self._db_with_hashes(tmpdir, hashes)
            index = SearchDistanceIndex.build(db)
            self.assertIsNotNone(index)
            n = index.num_frames
            peak = estimated_peak_bytes(n)
            steady = n * _BYTES_PER_FRAME + _OFFSETS_BYTES
            self.assertLess(steady, peak)
            self.assertEqual(peak, n * (_BYTES_PER_FRAME + 16) + _OFFSETS_BYTES)

            old_cap = fast_search_mod.MAX_INDEX_BYTES
            old_env = os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
            try:
                # Steady-state arrays fit, but the argsort scratch would exceed the cap.
                fast_search_mod.MAX_INDEX_BYTES = (steady + peak) // 2
                self.assertGreaterEqual(fast_search_mod.MAX_INDEX_BYTES, steady)
                self.assertLess(fast_search_mod.MAX_INDEX_BYTES, peak)
                self.assertIsNone(SearchDistanceIndex.build(db))

                fast_search_mod.MAX_INDEX_BYTES = peak - 1
                self.assertIsNone(SearchDistanceIndex.build(db))

                # Equal to the estimated peak is allowed; the index must be identical.
                fast_search_mod.MAX_INDEX_BYTES = peak
                rebuilt = SearchDistanceIndex.build(db)
                self.assertIsNotNone(rebuilt)
                self._assert_indexes_identical(rebuilt, index)
            finally:
                fast_search_mod.MAX_INDEX_BYTES = old_cap
                if old_env is None:
                    os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
                else:
                    os.environ[HVD_FAST_SEARCH_MAX_GIB] = old_env
            db.close()

    def _capture_print_user(self):
        printed: list[str] = []

        def capture(msg, **kwargs):
            printed.append(msg)

        return printed, mock.patch("hydrusvideodeduplicator.dedup_util.print_user", side_effect=capture)

    def test_max_index_bytes_env(self):
        old_env = os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
        old_cap = fast_search_mod.MAX_INDEX_BYTES
        try:
            self.assertEqual(max_index_bytes(), fast_search_mod.MAX_INDEX_BYTES)
            os.environ[HVD_FAST_SEARCH_MAX_GIB] = "4"
            self.assertEqual(max_index_bytes(), 4 * _GIB)
            os.environ[HVD_FAST_SEARCH_MAX_GIB] = "2.5"
            self.assertEqual(max_index_bytes(), int(2.5 * _GIB))
            os.environ[HVD_FAST_SEARCH_MAX_GIB] = "0"
            self.assertEqual(max_index_bytes(), fast_search_mod.MAX_INDEX_BYTES)
            os.environ[HVD_FAST_SEARCH_MAX_GIB] = "nope"
            self.assertEqual(max_index_bytes(), fast_search_mod.MAX_INDEX_BYTES)
            fast_search_mod.MAX_INDEX_BYTES = 123
            os.environ.pop(HVD_FAST_SEARCH_MAX_GIB)
            self.assertEqual(max_index_bytes(), 123)
        finally:
            fast_search_mod.MAX_INDEX_BYTES = old_cap
            if old_env is None:
                os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
            else:
                os.environ[HVD_FAST_SEARCH_MAX_GIB] = old_env

    def test_over_cap_prints_once_and_search_still_works(self):
        hashes = [h for h in _make_hashes() if h]
        with TemporaryDirectory() as tmpdir:
            db = self._db_with_hashes(tmpdir, hashes)
            index = SearchDistanceIndex.build(db)
            self.assertIsNotNone(index)
            peak = estimated_peak_bytes(index.num_frames)
            hash_ids = [r[0] for r in db.execute("SELECT hash_id FROM shape_search_cache ORDER BY hash_id;")]
            original = vptree.VpTreeManager(db)
            expected = [original.search_file(h, max_hamming_distance=51) for h in hash_ids]

            old_cap = fast_search_mod.MAX_INDEX_BYTES
            old_env = os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
            reset_fast_search_fallback_warning()
            printed, patcher = self._capture_print_user()
            try:
                fast_search_mod.MAX_INDEX_BYTES = peak - 1
                with patcher:
                    tree = vptree.VpTreeManager(db)
                    self.assertFalse(tree.enable_fast_search_index())
                    self.assertFalse(tree.enable_fast_search_index())
                self.assertEqual(printed, [fast_search_cap_warning(peak, peak - 1)])
                got = [tree.search_file(h, max_hamming_distance=51) for h in hash_ids]
                self.assertEqual(got, expected)
            finally:
                fast_search_mod.MAX_INDEX_BYTES = old_cap
                if old_env is None:
                    os.environ.pop(HVD_FAST_SEARCH_MAX_GIB, None)
                else:
                    os.environ[HVD_FAST_SEARCH_MAX_GIB] = old_env
            db.close()

    def test_disabled_by_env_prints_nothing(self):
        hashes = [h for h in _make_hashes() if h]
        with TemporaryDirectory() as tmpdir:
            db = self._db_with_hashes(tmpdir, hashes)
            old = os.environ.get("HVD_FAST_SEARCH")
            reset_fast_search_fallback_warning()
            printed, patcher = self._capture_print_user()
            try:
                os.environ["HVD_FAST_SEARCH"] = "0"
                with patcher:
                    self.assertIsNone(SearchDistanceIndex.build(db))
                    tree = vptree.VpTreeManager(db)
                    self.assertFalse(tree.enable_fast_search_index())
                self.assertEqual(printed, [])
            finally:
                if old is None:
                    os.environ.pop("HVD_FAST_SEARCH", None)
                else:
                    os.environ["HVD_FAST_SEARCH"] = old
            db.close()

    def test_build_error_prints_warning_once(self):
        hashes = [h for h in _make_hashes() if h]
        with TemporaryDirectory() as tmpdir:
            db = self._db_with_hashes(tmpdir, hashes)
            reset_fast_search_fallback_warning()
            printed, patcher = self._capture_print_user()
            with mock.patch.object(SearchDistanceIndex, "_allocate_arrays", side_effect=MemoryError("out of memory")):
                with patcher:
                    self.assertIsNone(SearchDistanceIndex.build(db))
                    self.assertIsNone(SearchDistanceIndex.build(db))
            self.assertEqual(len(printed), 1)
            self.assertEqual(printed[0], fast_search_build_error_warning(MemoryError("out of memory")))
            db.close()


if __name__ == "__main__":
    unittest.main()
