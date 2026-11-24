"""Tests for the review fixes: bounded fast search scratch, unique pair count, HTTP timeouts, tree loop guard."""

from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests

from hydrusvideodeduplicator import dedup, hydrus_api
from hydrusvideodeduplicator.db import DedupeDB, vptree
from hydrusvideodeduplicator.db import fast_search as fast_search_mod
from hydrusvideodeduplicator.db.fast_search import SearchDistanceIndex, fast_search_enabled

from .test_fast_search import _blob, _frame, _make_hashes
from .test_scan_search import _add


@unittest.skipUnless(fast_search_enabled(), "numpy with bitwise_count not available")
class TestBoundedQueryScratch(unittest.TestCase):
    def test_chunked_results_equal_unchunked(self):
        rng = random.Random(7)
        repeated = _frame(rng)
        hashes = [h for h in _make_hashes(seed=11) if h]
        # Frames repeated across many nodes (listed by all 16 tables) and a long query: every chunked path runs.
        hashes += [_blob([repeated] * rng.randint(1, 5) + [_frame(rng)]) for _ in range(40)]
        hashes.append(_blob([repeated] * 300 + [_frame(rng) for _ in range(50)]))
        index = SearchDistanceIndex(list(range(len(hashes))), hashes)
        expected = [index.sparse_distances_for(q) for q in hashes]
        with (
            mock.patch.object(fast_search_mod, "_QUERY_FRAME_CHUNK", 7),
            mock.patch.object(fast_search_mod, "_CANDIDATE_CHUNK", 64),
            mock.patch.object(fast_search_mod, "_MAX_PENDING_HITS", 5),
        ):
            for q, (positions, distances) in zip(hashes, expected):
                got_positions, got_distances = index.sparse_distances_for(q)
                self.assertEqual(got_positions.tolist(), positions.tolist())
                self.assertEqual(got_distances.tolist(), distances.tolist())
        for q in hashes[-45:]:
            self.assertEqual(index.distances_for(q), [vptree.calculate_distance(q, h) for h in hashes])


class _FakeDb:
    def get_file_hash(self, hash_id):
        return f"{hash_id:064x}"

    def commit(self):
        pass


class _FakeTree:
    # file 1 finds 2 and 3; file 2 doesn't find 1 (a one-way match); file 3 finds 1 back
    results = {1: [(1, 0), (2, 10), (3, 5)], 2: [(2, 0)], 3: [(3, 0), (1, 5)]}

    def search_file(self, hash_id, max_hamming_distance):
        return self.results[hash_id]


class TestUniquePairCount(unittest.TestCase):
    def _deduper(self):
        d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        d.db = _FakeDb()
        d.update_progress_callback = None
        d.should_skip_step_fn = None
        d.dry_run = False
        d._update_search_cache = lambda *args: None
        return d

    def test_one_way_matches_are_counted(self):
        files = [(1,), (2,), (3,)]
        for name in ("_find_potential_duplicates_serial", "_find_potential_duplicates_pipelined"):
            sent = []
            count = getattr(self._deduper(), name)(_FakeTree(), files, 50, lambda a, b: sent.append((a, b)))
            self.assertEqual(len(sent), 3, name)  # same Hydrus calls as before
            self.assertEqual(count, 2, name)  # {1, 2} and {1, 3}; the old directional // 2 gave 1

    def test_unordered_pair(self):
        self.assertEqual(dedup.unordered_pair("b", "a"), dedup.unordered_pair("a", "b"))


class _RecordingSession:
    def __init__(self, error=None):
        self.kwargs = None
        self.error = error

    def request(self, method, url, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"version": 1}'
        return response


class TestHydrusTimeouts(unittest.TestCase):
    def test_default_timeout_is_passed_to_requests(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HVD_HYDRUS_TIMEOUT", None)
            session = _RecordingSession()
            hydrus_api.Client("key", session=session).get_api_version()
            self.assertEqual(session.kwargs["timeout"], (hydrus_api.CONNECT_TIMEOUT, hydrus_api.DEFAULT_READ_TIMEOUT))

    def test_env_override_and_explicit_timeout(self):
        for value, read in (("30", 30.0), ("0", None)):
            with mock.patch.dict(os.environ, {"HVD_HYDRUS_TIMEOUT": value}):
                session = _RecordingSession()
                hydrus_api.Client("key", session=session).get_api_version()
                self.assertEqual(session.kwargs["timeout"], (hydrus_api.CONNECT_TIMEOUT, read))
        session = _RecordingSession()
        hydrus_api.Client("key", session=session, timeout=(1, 2)).get_api_version()
        self.assertEqual(session.kwargs["timeout"], (1, 2))

    def test_timeout_becomes_a_hydrus_connection_error(self):
        session = _RecordingSession(error=requests.ReadTimeout("stalled"))
        with self.assertRaises(hydrus_api.ConnectionError):
            hydrus_api.Client("key", session=session).get_api_version()


class TestDamagedTreeGuard(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _looped_db(self, tmpdir):
        db = DedupeDB.DedupeDb(Path(tmpdir), "loop.sqlite")
        db.init_connection()
        db.create_tables()
        _add(db, [h for h in _make_hashes(seed=3) if h][:30])
        db.begin_transaction()
        with db.conn:
            db.relink_search_tree()
        root = db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()[0]
        leaf_query = "SELECT phash_id FROM shape_vptree WHERE inner_id IS NULL AND outer_id IS NULL;"
        last = db.execute(leaf_query).fetchone()[0]
        db.execute(
            "UPDATE shape_vptree SET radius = ?, inner_id = ? WHERE phash_id = ?;", (vptree.MAX_DISTANCE, root, last)
        )
        db.commit()
        return db

    def test_search_raises_instead_of_spinning(self):
        modes = ["original"] + (["index", "scan"] if fast_search_enabled() else [])
        with TemporaryDirectory() as tmpdir:
            db = self._looped_db(tmpdir)
            hash_id = db.execute("SELECT hash_id FROM shape_search_cache LIMIT 1;").fetchone()[0]
            for mode in modes:
                tree = vptree.VpTreeManager(db)
                if mode in ("index", "scan"):
                    self.assertTrue(tree.enable_fast_search_index())
                if mode == "scan":
                    self.assertTrue(tree.enable_fast_search_scan())
                with self.assertRaises(vptree.DamagedSearchTreeError, msg=mode):
                    tree.search_file(hash_id, max_hamming_distance=51)
            db.close()

    def test_add_leaf_raises_instead_of_spinning(self):
        new_hash = _blob([_frame(random.Random(99)) for _ in range(5)])
        for fast in ("0", "1"):
            with TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"HVD_FAST_INSERT": fast}):
                vptree.TemporaryIntegerTableNameCache()
                db = self._looped_db(tmpdir)
                phash_id = db.add_perceptual_hash(new_hash)
                with vptree.fast_tree_building(db):
                    with self.assertRaises(vptree.DamagedSearchTreeError, msg=f"HVD_FAST_INSERT={fast}"):
                        vptree.VpTreeManager(db).add_leaf(phash_id, new_hash)
                db.close()

    def test_normal_tree_is_not_flagged(self):
        with TemporaryDirectory() as tmpdir:
            db = DedupeDB.DedupeDb(Path(tmpdir), "ok.sqlite")
            db.init_connection()
            db.create_tables()
            _add(db, _make_hashes(seed=5))
            tree = vptree.VpTreeManager(db)
            for (hash_id,) in db.execute("SELECT hash_id FROM shape_search_cache;").fetchall():
                tree.search_file(hash_id, max_hamming_distance=101)
            db.close()


if __name__ == "__main__":
    unittest.main()
