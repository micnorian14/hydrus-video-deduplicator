from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator.db import DedupeDB, vptree
from hydrusvideodeduplicator.db.fast_search import SearchDistanceIndex, fast_search_enabled


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

    def test_can_be_disabled(self):
        old = os.environ.get("HVD_FAST_SEARCH")
        os.environ["HVD_FAST_SEARCH"] = "0"
        try:
            self.assertFalse(fast_search_enabled())
        finally:
            if old is None:
                del os.environ["HVD_FAST_SEARCH"]
            else:
                os.environ["HVD_FAST_SEARCH"] = old


if __name__ == "__main__":
    unittest.main()
