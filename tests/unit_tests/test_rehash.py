"""--rehash-overhashed: detection, and replacing the hashes through the normal database code paths."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from hydrusvideodeduplicator import dedup, rehash
from hydrusvideodeduplicator.db import vptree

from .test_fast_search import _blob as _frames_blob
from .test_fast_search import _frame
from .test_prune import _assert_valid_tree
from .test_relink import _new_db
from .test_tree_build_queue import _build_tree


def _blob(rng, frames: int) -> bytes:
    return _frames_blob([_frame(rng) for _ in range(frames)])


class TestIsOverhashed(unittest.TestCase):
    def test_threshold(self):
        self.assertTrue(rehash.is_overhashed(1578, 30_100))  # a WebM from the bug report: every frame hashed
        self.assertTrue(rehash.is_overhashed(30, 9_000))  # 3.3 per second
        self.assertFalse(rehash.is_overhashed(30, 10_000))  # exactly 3 per second is not over
        self.assertFalse(rehash.is_overhashed(29, 1_000))  # too few frames to judge (short clips, GIFs)
        self.assertFalse(rehash.is_overhashed(31, 30_000))  # normal: about one per second
        self.assertFalse(rehash.is_overhashed(45, 30_000))  # 1.5 per second (1.4 fps video, every frame)
        self.assertFalse(rehash.is_overhashed(5000, None))  # unknown length: never touched
        self.assertFalse(rehash.is_overhashed(5000, 0))

    def test_find_overhashed_batches(self):
        candidates = [(f"{i:064x}", 40 if i % 100 else 400) for i in range(600)]
        calls = []

        def durations(hashes):
            calls.append(len(hashes))
            return {h: 40_000 for h in hashes}

        found = rehash.find_overhashed(candidates, durations)
        self.assertEqual(calls, [256, 256, 88])
        self.assertEqual([h for h, _, _ in found], [f"{i:064x}" for i in range(0, 600, 100)])


class _Response:
    def __init__(self, content: bytes):
        self.content = content


class _ReadOnlyHydrus:
    """Only the two GET calls --rehash-overhashed may use. Any other call fails the test (AttributeError)."""

    def __init__(self, durations: dict):
        self.durations = durations
        self.calls = []

    def get_file_metadata(self, hashes, only_return_basic_information=None):
        self.calls.append(("get_file_metadata", len(hashes)))
        return {"metadata": [{"hash": h, "duration": self.durations.get(h)} for h in hashes]}

    def get_file(self, hash_=None):
        self.calls.append(("get_file", hash_))
        return _Response(hash_.encode())


class _Client:
    def __init__(self, durations):
        self.client = _ReadOnlyHydrus(durations)


class _FakePhash:
    def __init__(self, data: bytes):
        self.bytes = data


class TestRehashOverhashed(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def test_rehash_replaces_only_overhashed_files(self):
        import random

        rng = random.Random(3)
        hashes = [_blob(rng, rng.randint(1, 40)) for _ in range(30)]
        hashes[5] = _blob(rng, 900)  # over-hashed: 900 frames for 30 s
        hashes[9] = _blob(rng, 600)  # over-hashed, re-hashes to the same result
        hashes[12] = _blob(rng, 500)  # over-hashed, re-hashing fails
        hashes[20] = _blob(rng, 800)  # many frames but long enough (800 s): not touched
        file_hashes = [f"{i:064x}" for i in range(len(hashes))]
        durations = {h: 30_000 for h in file_hashes}
        durations[file_hashes[20]] = 800_000
        new_phash = {file_hashes[5]: _blob(rng, 30), file_hashes[9]: hashes[9]}

        def fake_compute(video, num_threads=0, gpu_decode=False):
            file_hash = video.decode()
            if file_hash == file_hashes[12]:
                raise RuntimeError("cannot decode")
            return _FakePhash(new_phash[file_hash])

        with TemporaryDirectory() as tmp:
            db = _new_db(tmp, "videohashes.sqlite", hashes)
            client = _Client(durations)
            with mock.patch.object(dedup, "compute_phash", fake_compute), open(os.devnull, "w") as null:
                with mock.patch("sys.stdout", null), mock.patch("sys.stderr", null):
                    result = rehash.run_rehash_overhashed(db, client, 1)

            self.assertEqual((result.found, result.rehashed, result.unchanged, result.failed), (3, 1, 1, 1))
            self.assertEqual((result.old_frames, result.new_frames), (900, 30))
            self.assertTrue(all(name in ("get_file_metadata", "get_file") for name, _ in client.client.calls))
            # The changed file waits in the queue with its new hash; its old hash left the tree.
            queue = db.execute("SELECT file_hash, phash FROM phashed_file_queue;").fetchall()
            self.assertEqual(queue, [(file_hashes[5], new_phash[file_hashes[5]])])
            self.assertIsNone(db.get_hash_id(file_hashes[5]))
            self.assertIsNone(db.get_phash_id(hashes[5]))
            for i in (9, 12, 20):  # unchanged, failed and long files keep their hash
                self.assertEqual(db.get_stored_phash(file_hashes[i]), hashes[i])
            self.assertEqual(db.check_search_tree()["problems"], 0)
            _assert_valid_tree(self, db)

            # The normal "Building search tree" step adds it back.
            self.assertEqual(_build_tree(db), "finished")
            self.assertEqual(db.get_stored_phash(file_hashes[5]), new_phash[file_hashes[5]])
            self.assertEqual(db.count_queued_files(), 0)
            self.assertEqual(db.check_search_tree()["problems"], 0)
            self.assertEqual(db.count_orphaned_files(), 0)
            _assert_valid_tree(self, db)
            self.assertEqual(db.get_num_phashed_files(), len(hashes))
            db.close()

    def test_nothing_found_changes_nothing(self):
        import random

        rng = random.Random(4)
        hashes = [_blob(rng, 35) for _ in range(5)]
        with TemporaryDirectory() as tmp:
            db = _new_db(tmp, "videohashes.sqlite", hashes)
            before = Path(tmp, "videohashes.sqlite").read_bytes()
            client = _Client({f"{i:064x}": 60_000 for i in range(5)})
            with open(os.devnull, "w") as null, mock.patch("sys.stdout", null), mock.patch("sys.stderr", null):
                result = rehash.run_rehash_overhashed(db, client, 1)
            self.assertEqual((result.found, result.rehashed), (0, 0))
            self.assertEqual([name for name, _ in client.client.calls], ["get_file_metadata"])
            db.close()
            self.assertEqual(Path(tmp, "videohashes.sqlite").read_bytes(), before)

    def test_queued_files_are_checked_too(self):
        import random

        rng = random.Random(5)
        with TemporaryDirectory() as tmp:
            db = _new_db(tmp, "videohashes.sqlite", [_blob(rng, 3)])
            db.add_to_phashed_files_queue("ab" * 32, _blob(rng, 200))
            db.commit()
            self.assertEqual(db.get_hashed_frame_counts(30), [("ab" * 32, 200)])
            new = _blob(rng, 10)
            db.begin_transaction()
            with db.conn:
                db.replace_perceptual_hashes({"ab" * 32: new})
            self.assertEqual(db.get_stored_phash("ab" * 32), new)
            self.assertEqual(db.count_queued_files(), 1)
            db.close()


if __name__ == "__main__":
    unittest.main()
