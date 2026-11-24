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

Memory: about 100 bytes per indexed frame (the frames themselves plus 16 int32 lookup tables) plus ~8 MB for
the block-value offsets. Construction never holds a second full-size copy of the frames: they are written
straight into the index, and the lookup tables are built from zero-copy views of those words. Peak construction
memory is about 116 bytes per frame plus ~8 MB (the extra is a temporary argsort buffer). The index is refused
if that peak would exceed MAX_INDEX_BYTES (2.5 GB by default; override with
HVD_FAST_SEARCH_MAX_GIB). Only nodes reachable from the tree root are indexed, because the
search can never visit any other node. The index is read-only, lives only for the duration of the search phase,
and never touches the database except for reading.

Disable with the environment variable HVD_FAST_SEARCH=0 to get the original behaviour.
"""

from __future__ import annotations

import logging
import math
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

# Refuse to build an index whose *peak construction* memory would exceed this. The search then uses the original code.
# Override with HVD_FAST_SEARCH_MAX_GIB (GiB, e.g. HVD_FAST_SEARCH_MAX_GIB=4). See max_index_bytes().
MAX_INDEX_BYTES = 2_500_000_000
HVD_FAST_SEARCH_MAX_GIB = "HVD_FAST_SEARCH_MAX_GIB"
_GIB = 1024**3
_fast_search_fallback_warned = False
_BYTES_PER_FRAME = FRAME_BYTES + 4 * _BLOCKS + 4  # frames + 16 int32 sorted-position tables + frame->node table
_OFFSETS_LEN = 65537
_OFFSETS_BYTES = _BLOCKS * _OFFSETS_LEN * 8  # 16 int64 offset tables, ~8.4 MB
# Stable argsort of one 16-bit column: int64 order (~8 bytes/frame) plus a mergesort temp (~8 bytes/frame).
_ARGSORT_SCRATCH_PER_FRAME = 16
_PEAK_BYTES_PER_FRAME = _BYTES_PER_FRAME + _ARGSORT_SCRATCH_PER_FRAME
# Largest distance (similarity 0), the distance of every node that shares no matching frame with the query.
MAX_DISTANCE = 101  # == vptree.MAX_DISTANCE == vptree.fix_vpdq_similarity(0.0)
# Bounds on per-query scratch memory (results are the same whatever these are):
_QUERY_FRAME_CHUNK = 1024  # query frames looked up at once (lookup arrays: ~6.7 MB per 1024 frames)
_CANDIDATE_CHUNK = 1 << 20  # candidate frames checked at once for one query frame (~25 MB of temporaries)
_MAX_PENDING_HITS = 1 << 22  # matched (frame, node) entries held before they are merged into per-node counts


def estimated_peak_bytes(total_frames: int) -> int:
    """Peak RAM during index construction: steady-state arrays + argsort scratch + fixed offset tables."""
    return total_frames * _PEAK_BYTES_PER_FRAME + _OFFSETS_BYTES


def max_index_bytes() -> int:
    """Peak construction RAM allowed for the fast search index, in bytes."""
    raw = os.environ.get(HVD_FAST_SEARCH_MAX_GIB, "").strip()
    if raw:
        try:
            gib = float(raw)
            if gib > 0:
                return int(gib * _GIB)
        except ValueError:
            pass
    return MAX_INDEX_BYTES


def format_index_gb(num_bytes: int) -> str:
    """User-facing size, in decimal GB (the default cap is 2.5 GB)."""
    text = f"{num_bytes / 1_000_000_000:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text} GB"


def suggested_max_gib(needed_bytes: int) -> int:
    """A HVD_FAST_SEARCH_MAX_GIB value large enough for needed_bytes."""
    return max(math.ceil(needed_bytes / _GIB), 1)


def fast_search_cap_warning(needed_bytes: int, cap_bytes: int) -> str:
    needed = format_index_gb(needed_bytes)
    cap = format_index_gb(cap_bytes)
    suggested = suggested_max_gib(needed_bytes)
    return (
        f"WARNING: Fast search is off: the index would need {needed}, over the {cap} cap. "
        f"Searches will be slower. Set {HVD_FAST_SEARCH_MAX_GIB}={suggested} to raise the cap."
    )


def fast_search_build_error_warning(exc: BaseException) -> str:
    return f"WARNING: Fast search is off: the index could not be built ({exc}). Searches will be slower."


def fast_search_numpy_warning() -> str:
    return "WARNING: Fast search is off: NumPy with bitwise_count is not available. Searches will be slower."


def fast_search_scan_error_warning(exc: BaseException) -> str:
    return f"WARNING: Fast search scan is off: it could not be prepared ({exc}). Searches will be slower."


def reset_fast_search_fallback_warning() -> None:
    """Allow the fallback warning to be printed again (for tests)."""
    global _fast_search_fallback_warned
    _fast_search_fallback_warned = False


def warn_fast_search_fallback(message: str) -> None:
    """Print one user-visible fallback warning per process. Logging is left to the caller."""
    global _fast_search_fallback_warned
    if _fast_search_fallback_warned:
        return
    _fast_search_fallback_warned = True
    try:
        from hydrusvideodeduplicator.dedup_util import print_user

        print_user(message, highlight=False)
    except Exception:
        try:
            print(message)
        except Exception:
            pass


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


def _usable_blob(blob) -> int | None:
    """Return the frame count if `blob` is a non-empty multiple of FRAME_BYTES, otherwise None."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    length = len(blob)
    if length == 0 or length % FRAME_BYTES:
        return None
    return length // FRAME_BYTES


class SearchDistanceIndex:
    """Read-only in-memory index of the tree nodes' frame hashes. Build with SearchDistanceIndex.build(db)."""

    def __init__(self, phash_ids: list[int], blobs: list[bytes]):
        ids: list[int] = []
        valid: list[bytes] = []
        total_frames = 0
        for pid, blob in zip(phash_ids, blobs):
            k = _usable_blob(blob)
            if k is None:
                continue
            ids.append(int(pid))
            valid.append(blob)
            total_frames += k
        self._allocate_arrays(total_frames)
        self._fill_from_blobs(valid)
        self._finish(ids, total_frames)

    def _allocate_arrays(self, total_frames: int) -> None:
        self._words = np.empty((4, total_frames), dtype=np.uint64)
        self._frame_owner = np.empty(total_frames, dtype=np.int32)

    def _write_frames(self, blob, cursor: int, node_idx: int, k: int) -> int:
        src = np.frombuffer(blob, dtype=np.uint64).reshape(k, 4).T
        self._words[:, cursor : cursor + k] = src
        self._frame_owner[cursor : cursor + k] = node_idx
        return cursor + k

    def _fill_from_blobs(self, blobs: list[bytes]) -> None:
        cursor = 0
        for node_idx, blob in enumerate(blobs):
            k = len(blob) // FRAME_BYTES
            cursor = self._write_frames(blob, cursor, node_idx, k)
        if cursor != self._words.shape[1]:
            raise ValueError("frame count mismatch while filling the search index")

    def _finish(self, phash_ids: list[int], total_frames: int) -> None:
        self._num_frames = total_frames
        self._num_nodes = len(phash_ids)
        self.position_of = {int(pid): i for i, pid in enumerate(phash_ids)}
        self._build_lookup_tables()

    def _build_lookup_tables(self) -> None:
        num_frames = self._num_frames
        self._perm = np.empty((_BLOCKS, num_frames), dtype=np.int32)
        self._offsets = np.empty((_BLOCKS, _OFFSETS_LEN), dtype=np.int64)
        for w in range(4):
            # self._words[w] is a 1D contiguous uint64 array; viewing it as uint16 yields the 4 blocks of word w.
            # block j = 4 * w + sub is a zero-copy view, identical to frames.view(uint16).reshape(N, 16)[:, j].
            block_group = self._words[w].view(np.uint16).reshape(num_frames, 4)
            for sub in range(4):
                j = 4 * w + sub
                column = block_group[:, sub]
                # kind="stable" keeps self._perm bit-identical to the previous construction path.
                order = np.argsort(column, kind="stable")
                self._perm[j] = order
                # frames with block j == v are self._perm[j][self._offsets[j, v] : self._offsets[j, v + 1]]
                counts = np.bincount(column, minlength=65536)
                self._offsets[j, 0] = 0
                np.cumsum(counts, out=self._offsets[j, 1:])
                del order
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
            if os.environ.get("HVD_FAST_SEARCH", "1").strip() != "0":
                log.warning("Fast search requires NumPy with bitwise_count; using the original search code.")
                warn_fast_search_fallback(fast_search_numpy_warning())
            return None
        try:
            start = time.perf_counter()
            reachable = _reachable_phash_ids(db)
            if not reachable:
                return None
            # Pass 1: measure usable reachable hashes without loading any blob data into Python.
            phash_ids: list[int] = []
            frame_counts: list[int] = []
            total_frames = 0
            cap = max_index_bytes()
            over_cap = False
            for phash_id, typ, length in db.execute(
                "SELECT phash_id, typeof(phash), length(phash) FROM shape_perceptual_hashes;"
            ):
                if phash_id not in reachable:
                    continue
                if typ != "blob" or length is None or length <= 0 or length % FRAME_BYTES:
                    continue
                k = length // FRAME_BYTES
                total_frames += k
                if over_cap:
                    continue
                phash_ids.append(int(phash_id))
                frame_counts.append(k)
                if estimated_peak_bytes(total_frames) > cap:
                    over_cap = True
            if over_cap:
                needed = estimated_peak_bytes(total_frames)
                log.warning("Fast search index would be too large; using the original search code.")
                warn_fast_search_fallback(fast_search_cap_warning(needed, cap))
                return None
            if not phash_ids:
                return None
            expected = {pid: k for pid, k in zip(phash_ids, frame_counts)}
            # Pass 2: stream each blob straight into the pre-allocated index arrays (no blobs list, no join copy).
            index = cls.__new__(cls)
            index._allocate_arrays(total_frames)
            cursor = 0
            node_idx = 0
            for phash_id, phash in db.execute("SELECT phash_id, phash FROM shape_perceptual_hashes;"):
                k = expected.get(phash_id)
                if k is None:
                    continue
                if not isinstance(phash, (bytes, bytearray, memoryview)) or len(phash) != k * FRAME_BYTES:
                    raise ValueError(f"phash_id {phash_id} changed during fast search index construction")
                cursor = index._write_frames(phash, cursor, node_idx, k)
                node_idx += 1
            if cursor != total_frames or node_idx != len(phash_ids):
                raise ValueError("shape_perceptual_hashes changed during fast search index construction")
            index._finish(phash_ids, total_frames)
            log.info(
                f"Fast search index: {index.num_nodes} nodes, {index.num_frames} frames, "
                f"{index.nbytes() / 1e6:.0f} MB, built in {time.perf_counter() - start:.1f}s"
            )
            return index
        except Exception as exc:  # includes MemoryError: never let the speedup break a run
            log.warning(f"Could not build fast search index, using the original search code: {exc!r}")
            warn_fast_search_fallback(fast_search_build_error_warning(exc))
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
        variants = self._flips.shape[0]
        # Matched node positions, one entry per (query frame, node) match: each query frame lists a node at most
        # once. Counted (and, for very large queries, merged into counts as it grows) at the end.
        hit_chunks = []
        pending_hits = 0
        merged = None  # (positions, counts) of the hits merged so far, only used for very large queries
        # Query frames are looked up in chunks so the (frames, 16, 17) lookup arrays stay small for long videos.
        for first in range(0, num_query_frames, _QUERY_FRAME_CHUNK):
            chunk = q[first : first + _QUERY_FRAME_CHUNK]
            chunk_frames = chunk.shape[0]
            query_blocks = chunk.view(np.uint16).reshape(chunk_frames, _BLOCKS)
            # For every query frame, block and variant ("same" or one bit flipped): the slice of the sorted table.
            keys = (query_blocks[:, :, None] ^ self._flips[None, None, :]).astype(np.intp)  # (frames, 16, 17)
            starts = self._offsets[self._rows, keys].reshape(chunk_frames, -1)
            lengths = self._offsets[self._rows, keys + 1].reshape(chunk_frames, -1) - starts
            del keys
            totals = lengths.sum(axis=1).tolist()
            for f in range(chunk_frames):
                if totals[f] == 0:
                    continue
                nonempty = np.flatnonzero(lengths[f])
                slices = zip(
                    (nonempty // variants).tolist(), starts[f, nonempty].tolist(), lengths[f, nonempty].tolist()
                )
                query_words = [np.uint64(int(w)) for w in chunk[f]]
                if totals[f] <= _CANDIDATE_CHUNK:
                    candidates = np.concatenate([self._perm[j, a : a + n] for j, a, n in slices])
                    matched = self._matching_frames(candidates, query_words)
                    if len(matched) == 0:
                        continue
                    # each query frame counts at most once per node, like the C code
                    owners = np.unique(self._frame_owner[matched])
                else:
                    # Very many candidates (e.g. one frame repeated across the library; a frame can be listed by
                    # up to all 16 tables): check them in bounded pieces. Same frames are checked, so the same
                    # nodes match.
                    pieces = []
                    for j, a, n in slices:
                        for piece_start in range(a, a + n, _CANDIDATE_CHUNK):
                            piece = self._perm[j, piece_start : min(piece_start + _CANDIDATE_CHUNK, a + n)]
                            matched = self._matching_frames(piece, query_words)
                            if len(matched):
                                pieces.append(np.unique(self._frame_owner[matched]))
                    if not pieces:
                        continue
                    owners = np.unique(np.concatenate(pieces))
                    del pieces
                hit_chunks.append(owners)
                pending_hits += len(owners)
                if pending_hits > _MAX_PENDING_HITS:
                    merged = _merge_hit_counts(merged, hit_chunks)
                    hit_chunks = []
                    pending_hits = 0
        if hit_chunks or merged is not None:
            merged = _merge_hit_counts(merged, hit_chunks)
        if merged is None or len(merged[0]) == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        positions, matched_frames = merged
        similarity = (matched_frames.astype(np.int64) * 100) // num_query_frames  # exact integer floor(100 * m / n)
        return positions.astype(np.int64), (100 - similarity) + 1  # within [1, 101] because 0 <= similarity <= 100

    def _matching_frames(self, candidates, query_words):
        """
        The candidate frame indices at full 256-bit hamming distance < MATCH_TOLERANCE from the query frame.
        One 64-bit word at a time: a candidate is dropped as soon as the distance of the words seen so far is
        >= MATCH_TOLERANCE; the remaining words can only add to it, so exactly the frames with total distance
        < MATCH_TOLERANCE are kept.
        """
        words = self._words
        distance = None
        for w in range(4):
            bits = words[w].take(candidates)
            np.bitwise_xor(bits, query_words[w], out=bits)
            distance = np.bitwise_count(bits) if distance is None else distance + np.bitwise_count(bits)
            keep = np.flatnonzero(distance < MATCH_TOLERANCE)
            if len(keep) == 0:
                return candidates[:0]
            if len(keep) < len(candidates):
                candidates = candidates[keep]
                distance = distance[keep]
        return candidates


def _merge_hit_counts(merged, hit_chunks):
    """Fold per-query-frame node lists into (positions ascending, number of query frames matching each)."""
    if hit_chunks:
        positions, counts = np.unique(np.concatenate(hit_chunks), return_counts=True)
    else:
        positions = np.empty(0, dtype=np.int32)
        counts = np.empty(0, dtype=np.int64)
    if merged is None:
        return positions, counts.astype(np.int64)
    all_positions = np.concatenate([merged[0], positions])
    all_counts = np.concatenate([merged[1], counts.astype(np.int64)])
    positions, inverse = np.unique(all_positions, return_inverse=True)
    totals = np.zeros(len(positions), dtype=np.int64)
    np.add.at(totals, inverse, all_counts)
    return positions, totals
