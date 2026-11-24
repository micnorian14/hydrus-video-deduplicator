"""
Fast, exact distance index for the duplicate *search* phase.

Why this exists
---------------
During "Finding potential duplicates", VpTreeManager.search_perceptual_hashes() walks the vptree and, for every
node it touches, calls calculate_distance() -> vpdq.matchHashBytes(query, node, 31). That C call compares every
frame of the query video against every frame of the node video, so one search costs tens of thousands of full
video-vs-video comparisons and dominates the search phase.

This module computes the *same numbers* much faster. It never changes which nodes are visited, in what order,
or what is written to the database; it only replaces "ask the C function" with "look up a value that was
computed with the identical rule". Any node or query it cannot handle falls back to the original C call.

The rule it reproduces (verified against hvdaccelerators.vpdq.matchHashBytes, see scratch/fastsearch tests):
  * a vPDQ hash is a sequence of 32-byte (256-bit) frame hashes;
  * a query frame "matches" a node if ANY node frame is at hamming distance < 31 (strictly less) from it;
  * similarity = floor(100 * matching_query_frames / query_frames)   (0 for an empty query or empty node);
  * distance   = (100 - similarity) + 1                              (vptree.fix_vpdq_similarity).

How it is fast but still exact (multi-index hashing / pigeonhole principle)
----------------------------------------------------------------------------
Split each 256-bit frame into 16 blocks of 16 bits. If two frames differ in at most 30 bits, at least one of
the 16 blocks differs in at most 1 bit (if every block differed in >= 2 bits the total would be >= 32).
So for each query frame we only need the node frames whose block j equals the query's block j, or the query's
block j with one bit flipped, for some j. Those are found with 16 sorted lookup tables. Every candidate is then
checked with the full 256-bit hamming distance, exactly like the C code. No match can be missed and no
non-match can be counted, so the result is identical, not approximate.

Each query only does work for the frames that share a (near-)equal block with it: candidates are checked one
64-bit word at a time (a candidate is dropped as soon as its distance is already too large), and the result is
kept sparse (only the nodes with at least one matching frame; every other node is at the maximum distance 101).
Nothing per query is proportional to the size of the library.

Memory: about 100 bytes per indexed frame (the frames themselves plus 16 int32 lookup tables) plus ~8 MB.
Only nodes reachable from the tree root are indexed, because the search can never visit any other node.
The index is read-only, lives only for the duration of the search phase, and never touches the database
except for reading.

Disable with the environment variable HVD_FAST_SEARCH=0 to get the original behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a declared dependency, but never let this break the app
    np = None

if TYPE_CHECKING:
    from hydrusvideodeduplicator.db import DedupeDB

log = logging.getLogger("fast_search")
log.setLevel(logging.INFO)

# Must be kept equal to the tolerance used by vptree.calculate_distance().
MATCH_TOLERANCE = 31
FRAME_BYTES = 32  # one PDQ hash = 256 bits
_BLOCKS = 16  # 16 blocks of 16 bits; pigeonhole is valid because 16 * 2 = 32 > MATCH_TOLERANCE - 1 = 30

# Refuse to build an index bigger than this (estimated). The search then simply uses the original code.
MAX_INDEX_BYTES = 2_500_000_000
_BYTES_PER_FRAME = FRAME_BYTES + 4 * _BLOCKS + 4  # frames + 16 int32 sorted-position tables + frame->node table
# Largest distance (similarity 0), the distance of every node that shares no matching frame with the query.
MAX_DISTANCE = 101  # == vptree.MAX_DISTANCE == vptree.fix_vpdq_similarity(0.0)


def fast_search_enabled() -> bool:
    """True if the fast index can be used (numpy with bitwise_count present and not disabled by the user)."""
    if os.environ.get("HVD_FAST_SEARCH", "1").strip() == "0":
        return False
    return np is not None and hasattr(np, "bitwise_count")


def _reachable_phash_ids(db: DedupeDB.DedupeDb) -> set[int]:
    """phash_ids reachable from the tree root by following inner_id/outer_id, i.e. every node a search can visit."""
    children = {
        phash_id: (inner_id, outer_id)
        for (phash_id, inner_id, outer_id) in db.execute("SELECT phash_id, inner_id, outer_id FROM shape_vptree;")
    }
    # Same query the search itself uses to find the root.
    root = db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()
    seen: set[int] = set()
    if root is None:
        return seen
    stack = [root[0]]
    while stack:  # terminates: every id is expanded at most once (the 'seen' check guards against cycles)
        node = stack.pop()
        if node in seen or node not in children:
            continue
        seen.add(node)
        inner_id, outer_id = children[node]
        if inner_id is not None:
            stack.append(inner_id)
        if outer_id is not None:
            stack.append(outer_id)
    return seen


class SearchDistanceIndex:
    """Read-only in-memory index of the tree nodes' frame hashes. Build with SearchDistanceIndex.build(db)."""

    def __init__(self, phash_ids: list[int], blobs: list[bytes]):
        lens = np.fromiter((len(b) // FRAME_BYTES for b in blobs), dtype=np.int64, count=len(phash_ids))
        self._setup(phash_ids, lens, b"".join(blobs))

    def _setup(self, phash_ids: list[int], lens, joined: bytes):
        n = len(phash_ids)
        frames = np.frombuffer(joined, dtype=np.uint64).reshape(-1, 4)
        num_frames = frames.shape[0]
        self._num_frames = num_frames
        # The frames, one row per 64-bit word: self._words[w][i] is word w of frame i. Stored by word so a candidate
        # check can look at one word at a time and stop as soon as the hamming distance is already too large.
        self._words = np.ascontiguousarray(frames.T)
        # frame -> position of the node (video) it belongs to
        self._frame_owner = np.repeat(np.arange(n, dtype=np.int32), lens)
        self._num_nodes = n
        # phash_id -> position in the distance list returned by distances_for()
        self.position_of = {int(pid): i for i, pid in enumerate(phash_ids)}

        blocks = frames.view(np.uint16).reshape(num_frames, _BLOCKS)
        self._perm = np.empty((_BLOCKS, num_frames), dtype=np.int32)
        self._offsets = np.empty((_BLOCKS, 65537), dtype=np.int64)
        all_values = np.arange(65537)
        for j in range(_BLOCKS):
            column = blocks[:, j]
            order = np.argsort(column, kind="stable")
            self._perm[j] = order
            # frames with block j == v are self._perm[j][self._offsets[j, v] : self._offsets[j, v + 1]]
            self._offsets[j] = np.searchsorted(column[order], all_values)
            del order
        del blocks, frames
        self._flips = np.array([0] + [1 << k for k in range(16)], dtype=np.uint16)  # "same" + 16 one-bit flips
        self._rows = np.arange(_BLOCKS)[None, :, None]  # block index, broadcast over (frames, blocks, variants)

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def nbytes(self) -> int:
        return self._words.nbytes + self._perm.nbytes + self._offsets.nbytes + self._frame_owner.nbytes

    @classmethod
    def build(cls, db: DedupeDB.DedupeDb) -> SearchDistanceIndex | None:
        """Build the index, or return None (=> use the original code) if it can't or shouldn't be built."""
        if not fast_search_enabled():
            return None
        try:
            start = time.perf_counter()
            reachable = _reachable_phash_ids(db)
            if not reachable:
                return None
            phash_ids: list[int] = []
            blobs: list[bytes] = []
            total_bytes = 0
            for phash_id, phash in db.execute("SELECT phash_id, phash FROM shape_perceptual_hashes;"):
                if phash_id not in reachable:
                    continue
                if not isinstance(phash, (bytes, bytearray)) or len(phash) == 0 or len(phash) % FRAME_BYTES:
                    continue  # unusual hash: not indexed, the original C call handles it
                phash_ids.append(phash_id)
                blobs.append(bytes(phash))
                total_bytes += len(phash)
                if (total_bytes // FRAME_BYTES) * _BYTES_PER_FRAME > MAX_INDEX_BYTES:
                    log.warning("Fast search index would be too large; using the original search code.")
                    return None
            if not phash_ids:
                return None
            # Same as cls(phash_ids, blobs), but the separate blobs are freed before the index arrays are made.
            lens = np.fromiter((len(b) // FRAME_BYTES for b in blobs), dtype=np.int64, count=len(phash_ids))
            joined = b"".join(blobs)
            del blobs
            index = cls.__new__(cls)
            index._setup(phash_ids, lens, joined)
            del joined
            log.info(
                f"Fast search index: {index.num_nodes} nodes, {index.num_frames} frames, "
                f"{index.nbytes() / 1e6:.0f} MB, built in {time.perf_counter() - start:.1f}s"
            )
            return index
        except Exception as exc:  # includes MemoryError: never let the speedup break a run
            log.warning(f"Could not build fast search index, using the original search code: {exc!r}")
            return None

    def distances_for(self, query: bytes) -> list[int] | None:
        """
        Distance from `query` to every indexed node, identical to vptree.calculate_distance(query, node_phash).
        Indexed by self.position_of[phash_id]. Returns None for a query it does not handle (caller falls back).
        """
        distances = self.distance_array_for(query)
        return None if distances is None else distances.tolist()

    def distance_array_for(self, query: bytes):
        """Same values as distances_for(), as a numpy int64 array (or None => caller falls back)."""
        sparse = self.sparse_distances_for(query)
        if sparse is None:
            return None
        positions, distances = sparse
        result = np.full(self._num_nodes, MAX_DISTANCE, dtype=np.int64)
        result[positions] = distances
        return result

    def sparse_distances_for(self, query: bytes):
        """
        The same distances as distance_array_for(), but only for the nodes that share at least one matching frame
        with the query: (positions, distances), two numpy int64 arrays, positions sorted ascending. Every node that
        is not listed has no matching frame, so its distance is MAX_DISTANCE (similarity 0). Returns None for a
        query it does not handle (caller falls back). The work done is proportional to the frames that share a
        16-bit block with the query, never to the size of the library.
        """
        if not isinstance(query, (bytes, bytearray)) or len(query) == 0 or len(query) % FRAME_BYTES:
            return None
        try:
            return self._sparse_distances(query)
        except Exception as exc:  # e.g. MemoryError on a huge query: use the original code for this query
            log.warning(f"Fast search index failed for one query, using the original code for it: {exc!r}")
            return None

    def _sparse_distances(self, query: bytes):
        q = np.frombuffer(query, dtype=np.uint64).reshape(-1, 4)
        num_query_frames = q.shape[0]
        query_blocks = q.view(np.uint16).reshape(num_query_frames, _BLOCKS)
        # For every query frame, block and variant ("same" or one bit flipped): the slice of the sorted table.
        keys = (query_blocks[:, :, None] ^ self._flips[None, None, :]).astype(np.intp)  # (frames, 16, 17)
        starts = self._offsets[self._rows, keys].reshape(num_query_frames, -1)
        lengths = self._offsets[self._rows, keys + 1].reshape(num_query_frames, -1) - starts
        variants = self._flips.shape[0]
        words = self._words
        hit_keys = []  # per query frame: node position * num_query_frames + frame, for every node it matches
        for f in range(num_query_frames):
            nonempty = np.flatnonzero(lengths[f])
            if len(nonempty) == 0:
                continue
            candidates = np.concatenate(
                [
                    self._perm[j, a : a + n]
                    for j, a, n in zip(
                        (nonempty // variants).tolist(), starts[f, nonempty].tolist(), lengths[f, nonempty].tolist()
                    )
                ]
            )
            # Full 256-bit hamming distance, one 64-bit word at a time. A candidate is dropped as soon as the
            # distance of the words seen so far is >= MATCH_TOLERANCE; the remaining words can only add to it,
            # so exactly the frames with total distance < MATCH_TOLERANCE are kept.
            query_words = [np.uint64(int(w)) for w in q[f]]
            distance = None
            for w in range(4):
                bits = words[w].take(candidates)
                np.bitwise_xor(bits, query_words[w], out=bits)
                distance = np.bitwise_count(bits) if distance is None else distance + np.bitwise_count(bits)
                keep = np.flatnonzero(distance < MATCH_TOLERANCE)
                if len(keep) == 0:
                    break
                if len(keep) < len(candidates):
                    candidates = candidates[keep]
                    distance = distance[keep]
            else:
                # each query frame counts at most once per node, like the C code
                owners = np.unique(self._frame_owner[candidates]).astype(np.int64)
                hit_keys.append(owners * num_query_frames + f)
        if not hit_keys:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        positions, matched_frames = np.unique(np.concatenate(hit_keys) // num_query_frames, return_counts=True)
        similarity = (matched_frames.astype(np.int64) * 100) // num_query_frames  # exact integer floor(100 * m / n)
        return positions.astype(np.int64), (100 - similarity) + 1  # within [1, 101] because 0 <= similarity <= 100
