from __future__ import annotations

import collections
import contextlib
import logging
import os
import random
import sqlite3
from typing import TYPE_CHECKING

from hvdaccelerators import vpdq

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a declared dependency; without it the fast scan is just not used
    np = None

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from hydrusvideodeduplicator.db import DedupeDB

log = logging.getLogger("vptree")
log.setLevel(logging.INFO)

# Pretty much the entirety of this vptree is from Hydrus ClientDBSimilarFiles. Credit to Hydrus's developer.


def fix_vpdq_similarity(similarity: float) -> int:
    """Turn [100.0, 0.0] similarity to [1, 101]"""
    # TODO: We have to adjust to + 1 because phashes are rarely identical and there's some weird logic surrounding it.  # noqa: E501
    result = (100 - int(similarity)) + 1
    assert result > 0
    return result


# NOTE: This is the equivalent of HydrusData.Get64BitHammingDistance
def calculate_distance(phash_a: bytes, phash_b: bytes) -> int:
    """Get the distance between two perceptual hashes, from [1, 101], where 1 is very similar and 100 is not similar."""
    return fix_vpdq_similarity(vpdq.matchHashBytes(phash_a, phash_b, 31))


# Largest value calculate_distance() can return (similarity 0 -> distance 101).
MAX_DISTANCE = fix_vpdq_similarity(0.0)


def chain_tree_rows(perceptual_hash_ids: list[int]) -> list[tuple]:
    """
    shape_vptree rows that link the given phash_ids into one chain: each node's only child is the next id.

    Why this is a valid tree: every non-last node gets radius MAX_DISTANCE and its child as inner child. The inner
    side of a node means "distance <= radius", which is always true when radius is the largest possible distance,
    so every node really does belong inside all of its ancestors. search_perceptual_hashes() therefore always
    descends (101 <= radius + search radius) and checks every node, and add_leaf() (fast or original) always goes
    "inside" and appends new files at the end of the chain. The last node is a normal leaf (radius NULL).
    For this distance metric almost every pair of unrelated videos is at distance 101, so trees built one file at
    a time already end up as (mostly) this same chain; this just builds it directly and completely.
    Row layout: (phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population).
    """
    rows = []
    count = len(perceptual_hash_ids)
    for position, phash_id in enumerate(perceptual_hash_ids):
        is_last = position == count - 1
        rows.append(
            (
                phash_id,
                perceptual_hash_ids[position - 1] if position > 0 else None,
                None if is_last else MAX_DISTANCE,
                None if is_last else perceptual_hash_ids[position + 1],
                count - position - 1,  # inner_population = everything below this node
                None,
                0,
            )
        )
    return rows


class TemporaryIntegerTableNameCache:
    my_instance = None

    def __init__(self):
        TemporaryIntegerTableNameCache.my_instance = self

        self._column_names_to_table_names = collections.defaultdict(collections.deque)
        self._column_names_counter = collections.Counter()

    @staticmethod
    def instance() -> TemporaryIntegerTableNameCache:
        if TemporaryIntegerTableNameCache.my_instance is None:
            raise Exception("TemporaryIntegerTableNameCache is not yet initialised!")

        else:
            return TemporaryIntegerTableNameCache.my_instance

    def get_name(self, column_name):
        table_names = self._column_names_to_table_names[column_name]

        initialised = True

        if len(table_names) == 0:
            initialised = False

            i = self._column_names_counter[column_name]

            table_name = "mem.temp_int_{}_{}".format(column_name, i)

            table_names.append(table_name)

            self._column_names_counter[column_name] += 1

        table_name = table_names.pop()

        return (initialised, table_name)

    def release_name(self, column_name, table_name):
        self._column_names_to_table_names[column_name].append(table_name)


class TemporaryIntegerTable(object):
    def __init__(self, cursor: sqlite3.Cursor, integer_iterable, column_name):
        if not isinstance(integer_iterable, set):
            integer_iterable = set(integer_iterable)

        self._cursor = cursor
        self._integer_iterable = integer_iterable
        self._column_name = column_name

        (self._initialised, self._table_name) = TemporaryIntegerTableNameCache.instance().get_name(self._column_name)

    def __enter__(self):
        if not self._initialised:
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS {} ( {} INTEGER PRIMARY KEY );".format(self._table_name, self._column_name)
            )

        self._cursor.executemany(
            "INSERT INTO {} ( {} ) VALUES ( ? );".format(self._table_name, self._column_name),
            ((i,) for i in self._integer_iterable),
        )

        return self._table_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cursor.execute("DELETE FROM {};".format(self._table_name))

        TemporaryIntegerTableNameCache.instance().release_name(self._column_name, self._table_name)

        return False


def dedupe_list(xs: Iterable):
    if isinstance(xs, set):
        return list(xs)

    xs_seen = set()

    xs_return = []

    for x in xs:
        if x in xs_seen:
            continue

        xs_return.append(x)

        xs_seen.add(x)

    return xs_return


def build_key_to_list_dict(pairs):
    d = collections.defaultdict(list)

    for key, value in pairs:
        d[key].append(value)

    return d


# ---------------------------------------------------------------------------------------------------------------
# Fast tree building (used only by Dedup.process_phashed_file_queue, see fast_tree_building()).
#
# add_leaf() walks from the root down to where the new file belongs. On large trees that walk visits a huge number
# of nodes per file, and for each one the original code does a SELECT, a full vPDQ comparison, and afterwards one
# UPDATE per visited node for the population counters. The fast path makes the same decisions and the same writes:
#   * node rows are read once from the DB and then kept in memory (and updated in memory exactly as the DB is);
#   * distances come from the exact SearchDistanceIndex (db/fast_search.py), falling back to calculate_distance;
#   * the population UPDATEs are sent with executemany (same statements, same order, less Python overhead);
#   * all of one file's tree writes happen inside a SAVEPOINT, so an interruption (Ctrl+C) can never leave a
#     half-inserted file in the tree; the tree is then exactly as it was before that file.
# The in-memory copy is only valid while nothing else modifies shape_vptree, which is true during that phase.
# Set the environment variable HVD_FAST_INSERT=0 to use the original code.
# ---------------------------------------------------------------------------------------------------------------

_active_fast_insert = None  # the _FastInsertState in use, or None


def fast_insert_enabled() -> bool:
    return os.environ.get("HVD_FAST_INSERT", "1").strip() != "0"


# ---------------------------------------------------------------------------------------------------------------
# Fast search scan (duplicate search phase only)
#
# search_perceptual_hashes() walks the tree from the root in Python, one node per step. After a relink the tree is
# one long chain, so every search steps through ~all nodes, and that walk (not the distance math) is most of the
# search time. The fast scan replaces only the part of the walk whose outcome cannot depend on the query:
#
#   A node with radius R, an inner child and no outer child sends the walk to its inner child unless
#   distance > R + search_radius. Every distance is between 1 and MAX_DISTANCE (101), so when
#   R + search_radius >= MAX_DISTANCE the walk ALWAYS continues to the inner child, for every query.
#
# Starting at the root, the longest run of such nodes (the "prefix") is therefore visited in exactly the same
# order by every search with that radius. For the prefix we take all distances from the exact SearchDistanceIndex
# at once (numpy) and record the nodes with distance <= search_radius in prefix order, exactly as the walk would.
# The walk then continues from the node after the prefix with the original, unchanged code, so any node that
# could prune (other radii, outer children, missing or unindexed nodes, loops) is still handled by the original
# logic. Results (which files, which distances, what order) are identical; nothing is written to the database.
# Set the environment variable HVD_SCAN_SEARCH=0 to use the original walk.
# ---------------------------------------------------------------------------------------------------------------


def scan_search_enabled() -> bool:
    return os.environ.get("HVD_SCAN_SEARCH", "1").strip() != "0"


class _FastInsertState:
    def __init__(self, db: DedupeDB.DedupeDb):
        self.db = db
        # phash_id -> [phash, radius, inner_id, inner_population, outer_id, outer_population], same as the DB row
        self.nodes = {}
        self.distance_index = None

    def get_node(self, phash_id):
        node = self.nodes.get(phash_id)
        if node is None:
            row = self.db.execute(
                "SELECT phash, radius, inner_id, inner_population, outer_id, outer_population FROM shape_perceptual_hashes NATURAL JOIN shape_vptree WHERE phash_id = ?;",  # noqa: E501
                (phash_id,),
            ).fetchone()
            if row is None:
                return None  # not cached, exactly like the original (which re-queries every time)
            node = self.nodes[phash_id] = list(row)
        return node


@contextlib.contextmanager
def fast_tree_building(db: DedupeDB.DedupeDb):
    """While active, add_leaf() calls on `db` use the fast path. Yields True if the fast path is in use."""
    global _active_fast_insert
    state = None
    if fast_insert_enabled():
        state = _FastInsertState(db)
        from hydrusvideodeduplicator.db.fast_search import SearchDistanceIndex

        state.distance_index = SearchDistanceIndex.build(db)  # None => use calculate_distance for everything
    _active_fast_insert = state
    try:
        yield state is not None
    finally:
        _active_fast_insert = None


class VpTreeManager:
    TemporaryIntegerTableNameCache()

    def __init__(self, db: DedupeDB.DedupeDb):
        self.db = db
        try:
            # need this for the TemporaryIntegerTableNameCache
            # don't run this more than once. do this once per program run. this is a singleton.
            # TODO: move this somewhere appropriate. this is bad.
            cur = self.db.conn.cursor()
            cur.execute('ATTACH ":memory:" as mem')
        except sqlite3.OperationalError as exc:
            if str(exc) != "database mem is already in use":
                raise

        self._perceptual_hash_id_to_vp_tree_node_cache = {}
        self._non_vp_treed_perceptual_hash_ids = set()
        self._root_node_perceptual_hash_id = None
        self._reported_on_a_broken_branch = False
        # Optional in-memory index used only to compute search distances faster (see db/fast_search.py).
        # None means "use the original calculate_distance() for everything".
        self._search_distance_index = None
        # Tree shape for the fast search scan (see enable_fast_search_scan). None means "walk the whole tree".
        self._search_scan_nodes = None
        self._search_scan_prefixes = {}

    def enable_fast_search_index(self) -> bool:
        """
        Build the read-only fast distance index for search_file(). Only call this when the tree will not be
        modified while searching (the duplicate search phase). Distances are identical to calculate_distance();
        anything the index doesn't cover still uses calculate_distance(). Returns True if the index is in use.
        """
        from hydrusvideodeduplicator.db.fast_search import SearchDistanceIndex

        self._search_distance_index = SearchDistanceIndex.build(self.db)
        return self._search_distance_index is not None

    def enable_fast_search_scan(self) -> bool:
        """
        Use the fast search scan (see the comment above scan_search_enabled). Requires the fast search index and,
        like it, may only be used while the tree is not modified (the duplicate search phase). Returns True if used.
        """
        self._search_scan_nodes = None
        self._search_scan_prefixes = {}
        if self._search_distance_index is None or np is None or not scan_search_enabled():
            return False
        try:
            # Same join the search uses to load nodes, so "node exists" means exactly the same thing.
            self._search_scan_nodes = {
                phash_id: (radius, inner_id, outer_id)
                for (phash_id, radius, inner_id, outer_id) in self.db.execute(
                    "SELECT phash_id, radius, inner_id, outer_id FROM shape_perceptual_hashes CROSS JOIN shape_vptree USING ( phash_id );"  # noqa: E501
                )
            }
        except Exception as exc:  # never let the speedup break a run
            log.warning(f"Could not prepare the fast search scan, using the original search: {exc!r}")
            self._search_scan_nodes = None
            return False
        return True

    def _search_scan_prefix(self, search_radius):
        """
        (index positions, phash_ids, next phash_id, unindexed nodes) of the prefix of the walk that every search with
        this radius takes in the same order, or None if there is no such prefix. Cached per (root, radius).
        Unindexed nodes (e.g. an empty hash) are listed as (prefix position, radius, phash): their distance is computed
        with calculate_distance() for each search, and checked, exactly as the walk would do.
        """
        key = (self._root_node_perceptual_hash_id, search_radius)
        if key in self._search_scan_prefixes:
            return self._search_scan_prefixes[key]
        nodes = self._search_scan_nodes
        position_of = self._search_distance_index.position_of
        ids = []
        positions = []
        unindexed = []
        seen = set()
        node_id = self._root_node_perceptual_hash_id
        while node_id is not None and node_id not in seen:  # terminates: every node is taken at most once
            node = nodes.get(node_id)
            position = position_of.get(node_id)
            if node is None:
                break  # node missing from the tree tables: the original walk handles it
            radius, inner_id, outer_id = node
            if radius is None or inner_id is None or outer_id is not None or radius + search_radius < MAX_DISTANCE:
                break  # the walk could go differently here depending on the query: original walk from here
            if position is None:
                if len(unindexed) >= 1000:
                    break  # unusual: many unindexed nodes, let the original walk handle the rest
                (phash,) = self.db.execute(
                    "SELECT phash FROM shape_perceptual_hashes WHERE phash_id = ?;", (node_id,)
                ).fetchone()  # same value the walk's node cache holds
                unindexed.append((len(ids), radius, phash))
                position = 0  # placeholder; replaced by the calculate_distance() value for each search
            seen.add(node_id)
            ids.append(node_id)
            positions.append(position)
            node_id = inner_id
        prefix = (np.array(positions, dtype=np.intp), ids, node_id, unindexed) if ids else None
        self._search_scan_prefixes[key] = prefix
        return prefix

    def add_leaf(self, perceptual_hash_id, perceptual_hash):
        state = _active_fast_insert
        if state is not None and state.db is self.db:
            return self._add_leaf_fast(state, perceptual_hash_id, perceptual_hash)

        already_in_tree = self.db.execute(
            "SELECT 1 FROM shape_vptree WHERE phash_id = ?;", (perceptual_hash_id,)
        ).fetchone()

        if already_in_tree is not None:
            return

        result = self.db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()

        parent_id = None

        if result is not None:
            (root_node_perceptual_hash_id,) = result

            ancestors_we_are_inside = []
            ancestors_we_are_outside = []

            an_ancestor_is_unbalanced = False

            next_ancestor_id = root_node_perceptual_hash_id

            while next_ancestor_id is not None:
                ancestor_id = next_ancestor_id

                result = self.db.execute(
                    "SELECT phash, radius, inner_id, inner_population, outer_id, outer_population FROM shape_perceptual_hashes NATURAL JOIN shape_vptree WHERE phash_id = ?;",  # noqa: E501
                    (ancestor_id,),
                ).fetchone()

                if result is None:
                    if not self._reported_on_a_broken_branch:
                        # TODO: Update this message.
                        # TODO: Add CLI option to regenerate the vptree.
                        # Note: I hit this path once somehow due to hitting some exception and the program crashing, but
                        # it fixed itself on the next run. So maybe we should just log this and move on.
                        # message = "Hey, while trying to import a file, hydrus discovered a hole in the similar files search tree. Please run _database->regenerate->similar files search tree_ when it is convenient!"  # noqa: E501
                        # message += "\n" * 2
                        # message += "You will not see this message again this boot."
                        message = "Broken branch detected. Either restart the program and see if it goes away, or reset your dedupe tree using '--clear-search-tree'."  # noqa: E501

                        print(message)
                        log.error(message)
                        log.error(f"perceptual_hash: {perceptual_hash}\n")
                        log.error(f"phash_id: {perceptual_hash_id}\n")
                        log.error(f"\nnext_ancestor_id: {next_ancestor_id}\nancestor phash id: {ancestor_id}")

                        self._reported_on_a_broken_branch = True

                    # ok so there is a missing branch. typically from an import crash desync, is my best bet
                    # we still want to add our leaf because we need to add the file to the tree population, but we will add it to the ghost of the branch. no worries, the regen code will sort it all out  # noqa: E501
                    parent_id = ancestor_id

                    break

                (
                    ancestor_perceptual_hash,
                    ancestor_radius,
                    ancestor_inner_id,
                    ancestor_inner_population,
                    ancestor_outer_id,
                    ancestor_outer_population,
                ) = result

                distance_to_ancestor = calculate_distance(perceptual_hash, ancestor_perceptual_hash)

                if ancestor_radius is None or distance_to_ancestor <= ancestor_radius:
                    ancestors_we_are_inside.append(ancestor_id)
                    ancestor_inner_population += 1
                    next_ancestor_id = ancestor_inner_id

                    if ancestor_inner_id is None:
                        self.db.execute(
                            "UPDATE shape_vptree SET inner_id = ?, radius = ? WHERE phash_id = ?;",
                            (perceptual_hash_id, distance_to_ancestor, ancestor_id),
                        )

                        self.clear_perceptual_hashes_from_vptree_node_cache((ancestor_id,))

                        parent_id = ancestor_id

                else:
                    ancestors_we_are_outside.append(ancestor_id)
                    ancestor_outer_population += 1
                    next_ancestor_id = ancestor_outer_id

                    if ancestor_outer_id is None:
                        self.db.execute(
                            "UPDATE shape_vptree SET outer_id = ? WHERE phash_id = ?;",
                            (perceptual_hash_id, ancestor_id),
                        )

                        self.clear_perceptual_hashes_from_vptree_node_cache((ancestor_id,))

                        parent_id = ancestor_id

                if not an_ancestor_is_unbalanced and ancestor_inner_population + ancestor_outer_population > 16:
                    larger = max(ancestor_inner_population, ancestor_outer_population)
                    smaller = min(ancestor_inner_population, ancestor_outer_population)

                    if smaller / larger < 0.5:
                        self.db.execute(
                            "INSERT OR IGNORE INTO shape_maintenance_branch_regen ( phash_id ) VALUES ( ? );",
                            (ancestor_id,),
                        )

                        # we only do this for the eldest ancestor, as the eventual rebalancing will affect all children

                        an_ancestor_is_unbalanced = True

            for ancestor_id in ancestors_we_are_inside:
                self.db.execute(
                    "UPDATE shape_vptree SET inner_population = inner_population + 1 WHERE phash_id = ?;",
                    (ancestor_id,),
                )
            for ancestor_id in ancestors_we_are_outside:
                self.db.execute(
                    "UPDATE shape_vptree SET outer_population = outer_population + 1 WHERE phash_id = ?;",
                    (ancestor_id,),
                )

            self.clear_perceptual_hashes_from_vptree_node_cache(ancestors_we_are_inside)
            self.clear_perceptual_hashes_from_vptree_node_cache(ancestors_we_are_outside)

        radius = None
        inner_id = None
        inner_population = 0
        outer_id = None
        outer_population = 0

        self.db.execute(
            "INSERT OR REPLACE INTO shape_vptree ( phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population ) VALUES ( ?, ?, ?, ?, ?, ?, ? );",  # noqa: E501
            (perceptual_hash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population),
        )

        self.clear_perceptual_hashes_from_vptree_node_cache((perceptual_hash_id,))

    def _add_leaf_fast(self, state: _FastInsertState, perceptual_hash_id, perceptual_hash):
        """Same decisions and same resulting rows as add_leaf() above; see the 'Fast tree building' notes."""
        already_in_tree = self.db.execute(
            "SELECT 1 FROM shape_vptree WHERE phash_id = ?;", (perceptual_hash_id,)
        ).fetchone()

        if already_in_tree is not None:
            return

        result = self.db.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()

        parent_id = None
        ancestors_we_are_inside = []
        ancestors_we_are_outside = []
        writes = []  # (sql, params) issued during the walk in the original, in the same order
        node_edits = []  # (node, field index, value): in-memory copy of those writes, applied after they succeed

        if result is not None:
            (root_node_perceptual_hash_id,) = result

            fast_distances = None
            fast_positions = None
            if state.distance_index is not None:
                fast_distances = state.distance_index.distances_for(perceptual_hash)
                fast_positions = state.distance_index.position_of

            an_ancestor_is_unbalanced = False

            next_ancestor_id = root_node_perceptual_hash_id

            while next_ancestor_id is not None:
                ancestor_id = next_ancestor_id

                node = state.get_node(ancestor_id)

                if node is None:
                    if not self._reported_on_a_broken_branch:
                        message = "Broken branch detected. Either restart the program and see if it goes away, or reset your dedupe tree using '--clear-search-tree'."  # noqa: E501

                        print(message)
                        log.error(message)
                        log.error(f"perceptual_hash: {perceptual_hash}\n")
                        log.error(f"phash_id: {perceptual_hash_id}\n")
                        log.error(f"\nnext_ancestor_id: {next_ancestor_id}\nancestor phash id: {ancestor_id}")

                        self._reported_on_a_broken_branch = True

                    parent_id = ancestor_id

                    break

                (
                    ancestor_perceptual_hash,
                    ancestor_radius,
                    ancestor_inner_id,
                    ancestor_inner_population,
                    ancestor_outer_id,
                    ancestor_outer_population,
                ) = node

                distance_to_ancestor = None
                if fast_distances is not None:
                    fast_position = fast_positions.get(ancestor_id)
                    if fast_position is not None:
                        distance_to_ancestor = fast_distances[fast_position]
                if distance_to_ancestor is None:
                    distance_to_ancestor = calculate_distance(perceptual_hash, ancestor_perceptual_hash)

                if ancestor_radius is None or distance_to_ancestor <= ancestor_radius:
                    ancestors_we_are_inside.append(ancestor_id)
                    ancestor_inner_population += 1
                    next_ancestor_id = ancestor_inner_id

                    if ancestor_inner_id is None:
                        writes.append(
                            (
                                "UPDATE shape_vptree SET inner_id = ?, radius = ? WHERE phash_id = ?;",
                                (perceptual_hash_id, distance_to_ancestor, ancestor_id),
                            )
                        )
                        node_edits.append((node, 2, perceptual_hash_id))
                        node_edits.append((node, 1, distance_to_ancestor))

                        parent_id = ancestor_id

                else:
                    ancestors_we_are_outside.append(ancestor_id)
                    ancestor_outer_population += 1
                    next_ancestor_id = ancestor_outer_id

                    if ancestor_outer_id is None:
                        writes.append(
                            ("UPDATE shape_vptree SET outer_id = ? WHERE phash_id = ?;", (perceptual_hash_id, ancestor_id))
                        )
                        node_edits.append((node, 4, perceptual_hash_id))

                        parent_id = ancestor_id

                if not an_ancestor_is_unbalanced and ancestor_inner_population + ancestor_outer_population > 16:
                    larger = max(ancestor_inner_population, ancestor_outer_population)
                    smaller = min(ancestor_inner_population, ancestor_outer_population)

                    if smaller / larger < 0.5:
                        writes.append(
                            (
                                "INSERT OR IGNORE INTO shape_maintenance_branch_regen ( phash_id ) VALUES ( ? );",
                                (ancestor_id,),
                            )
                        )

                        an_ancestor_is_unbalanced = True

        # Write everything for this file as one unit. (The walk above never reads anything it would have written,
        # so doing the writes after the walk gives exactly the same rows as the original.)
        use_savepoint = self.db.conn.in_transaction  # a SAVEPOINT outside a transaction would commit on RELEASE
        if use_savepoint:
            self.db.execute("SAVEPOINT hvd_add_leaf;")
        try:
            for sql, params in writes:
                self.db.execute(sql, params)
            if ancestors_we_are_inside:
                self.db.cur.executemany(
                    "UPDATE shape_vptree SET inner_population = inner_population + 1 WHERE phash_id = ?;",
                    [(ancestor_id,) for ancestor_id in ancestors_we_are_inside],
                )
            if ancestors_we_are_outside:
                self.db.cur.executemany(
                    "UPDATE shape_vptree SET outer_population = outer_population + 1 WHERE phash_id = ?;",
                    [(ancestor_id,) for ancestor_id in ancestors_we_are_outside],
                )
            self.db.execute(
                "INSERT OR REPLACE INTO shape_vptree ( phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population ) VALUES ( ?, ?, ?, ?, ?, ?, ? );",  # noqa: E501
                (perceptual_hash_id, parent_id, None, None, 0, None, 0),
            )
        except BaseException:
            # Undo this file's partial tree writes (e.g. Ctrl+C), and forget the in-memory copy so it is re-read.
            state.nodes.clear()
            if use_savepoint:
                self.db.execute("ROLLBACK TO hvd_add_leaf;")
                self.db.execute("RELEASE hvd_add_leaf;")
            raise
        if use_savepoint:
            self.db.execute("RELEASE hvd_add_leaf;")

        # Mirror the writes in memory.
        for node, field, value in node_edits:
            node[field] = value
        for ancestor_id in ancestors_we_are_inside:
            state.nodes[ancestor_id][3] += 1
        for ancestor_id in ancestors_we_are_outside:
            state.nodes[ancestor_id][5] += 1

        self.clear_perceptual_hashes_from_vptree_node_cache(ancestors_we_are_inside)
        self.clear_perceptual_hashes_from_vptree_node_cache(ancestors_we_are_outside)
        self.clear_perceptual_hashes_from_vptree_node_cache((perceptual_hash_id,))

    def regenerate_tree(self):
        try:
            # TODO: Change log severity to debug.
            log.info("regenerating similar file search data")

            log.info("purging search info of orphans")

            self.db.execute("DELETE FROM shape_perceptual_hash_map WHERE hash_id NOT IN ( SELECT hash_id FROM files )")

            log.info("gathering all leaves")

            self.db.execute("DELETE FROM shape_vptree;")

            self._perceptual_hash_id_to_vp_tree_node_cache = {}
            self._non_vp_treed_perceptual_hash_ids = set()
            self._root_node_perceptual_hash_id = None

            all_nodes = self.db.execute("SELECT phash_id, phash FROM shape_perceptual_hashes;").fetchall()

            log.info(f"{len(all_nodes)} leaves found, now regenerating")

            (root_id, root_perceptual_hash) = self.pop_best_root_node(all_nodes)  # HydrusData.RandomPop( all_nodes )

            self.generate_branch(None, root_id, root_perceptual_hash, all_nodes)

            self.db.execute("DELETE FROM shape_maintenance_branch_regen;")

        finally:
            log.info("done regenerating tree!")

    def generate_branch(self, parent_id, perceptual_hash_id, perceptual_hash, children):
        process_queue = collections.deque()

        process_queue.append((parent_id, perceptual_hash_id, perceptual_hash, children))

        insert_rows = []

        num_done = 0
        num_to_do = len(children) + 1

        all_altered_phash_ids = set()

        while len(process_queue) > 0:
            log.info(f"generating new branch -- num_done: {num_done} num to do: {num_to_do}")

            (parent_id, perceptual_hash_id, perceptual_hash, children) = process_queue.popleft()

            if len(children) == 0:
                inner_id = None
                inner_population = 0

                outer_id = None
                outer_population = 0

                radius = None

            else:
                children = sorted(
                    (
                        (
                            calculate_distance(perceptual_hash, child_perceptual_hash),
                            child_id,
                            child_perceptual_hash,
                        )
                        for (child_id, child_perceptual_hash) in children
                    )
                )

                median_index = len(children) // 2

                median_radius = children[median_index][0]

                inner_children = [
                    (child_id, child_perceptual_hash)
                    for (distance, child_id, child_perceptual_hash) in children
                    if distance < median_radius
                ]
                radius_children = [
                    (child_id, child_perceptual_hash)
                    for (distance, child_id, child_perceptual_hash) in children
                    if distance == median_radius
                ]
                outer_children = [
                    (child_id, child_perceptual_hash)
                    for (distance, child_id, child_perceptual_hash) in children
                    if distance > median_radius
                ]

                if len(inner_children) <= len(outer_children):
                    radius = median_radius

                    inner_children.extend(radius_children)

                else:
                    radius = median_radius - 1

                    outer_children.extend(radius_children)

                inner_population = len(inner_children)
                outer_population = len(outer_children)

                (inner_id, inner_perceptual_hash) = self.pop_best_root_node(
                    inner_children
                )  # HydrusData.MedianPop( inner_children )

                if len(outer_children) == 0:
                    outer_id = None

                else:
                    (outer_id, outer_perceptual_hash) = self.pop_best_root_node(
                        outer_children
                    )  # HydrusData.MedianPop( outer_children )

            insert_rows.append(
                (perceptual_hash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population)
            )

            all_altered_phash_ids.add(perceptual_hash_id)

            if inner_id is not None:
                process_queue.append((perceptual_hash_id, inner_id, inner_perceptual_hash, inner_children))

            if outer_id is not None:
                process_queue.append((perceptual_hash_id, outer_id, outer_perceptual_hash, outer_children))

            num_done += 1

        log.info("branch constructed, now committing")

        for row in insert_rows:
            self.db.execute(
                "INSERT OR REPLACE INTO shape_vptree ( phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population ) VALUES ( ?, ?, ?, ?, ?, ?, ? );",  # noqa: E501
                row,
            )

        self.clear_perceptual_hashes_from_vptree_node_cache(all_altered_phash_ids)

    def pop_best_root_node(self, node_rows):
        if len(node_rows) == 1:
            root_row = node_rows.pop()

            return root_row

        MAX_VIEWPOINTS = 256
        MAX_SAMPLE = 64

        if len(node_rows) > MAX_VIEWPOINTS:
            viewpoints = random.sample(node_rows, MAX_VIEWPOINTS)

        else:
            viewpoints = node_rows

        if len(node_rows) > MAX_SAMPLE:
            sample = random.sample(node_rows, MAX_SAMPLE)

        else:
            sample = node_rows

        final_scores = []

        for v_id, v_perceptual_hash in viewpoints:
            views = sorted(
                (
                    calculate_distance(v_perceptual_hash, s_perceptual_hash)
                    for (s_id, s_perceptual_hash) in sample
                    if v_id != s_id
                )
            )

            # let's figure out the ratio of left_children to right_children, preferring 1:1, and convert it to a discrete integer score # noqa: E501

            median_index = len(views) // 2

            radius = views[median_index]

            num_left = len([1 for view in views if view < radius])
            num_radius = len([1 for view in views if view == radius])
            num_right = len([1 for view in views if view > radius])

            if num_left <= num_right:
                num_left += num_radius

            else:
                num_right += num_radius

            smaller = min(num_left, num_right)
            larger = max(num_left, num_right)

            ratio = smaller / larger

            ratio_score = int(ratio * MAX_SAMPLE / 2)

            # now let's calc the standard deviation--larger sd tends to mean less sphere overlap when searching

            mean_view = sum(views) / len(views)
            squared_diffs = [(view - mean_view) ** 2 for view in views]
            sd = (sum(squared_diffs) / len(squared_diffs)) ** 0.5

            final_scores.append((ratio_score, sd, v_id))

        final_scores.sort()

        # we now have a list like [ ( 11, 4.0, [id] ), ( 15, 3.7, [id] ), ( 15, 4.3, [id] ) ]

        (ratio_gumpf, sd_gumpf, root_id) = final_scores.pop()

        for i, (v_id, v_perceptual_hash) in enumerate(node_rows):
            if v_id == root_id:
                root_row = node_rows.pop(i)

                return root_row

    def clear_perceptual_hashes_from_vptree_node_cache(self, perceptual_hash_ids: Collection[int]):
        for perceptual_hash_id in perceptual_hash_ids:
            if perceptual_hash_id in self._perceptual_hash_id_to_vp_tree_node_cache:
                del self._perceptual_hash_id_to_vp_tree_node_cache[perceptual_hash_id]

            self._non_vp_treed_perceptual_hash_ids.discard(perceptual_hash_id)

            if self._root_node_perceptual_hash_id == perceptual_hash_id:
                self._root_node_perceptual_hash_id = None

    def _STL(self, iterable_cursor):
        # strip singleton tuples to a list

        return [item for (item,) in iterable_cursor]

    def _STS(self, iterable_cursor):
        # strip singleton tuples to a set

        return {item for (item,) in iterable_cursor}

    def _make_temporary_integer_table(self, integer_iterable, column_name):
        return TemporaryIntegerTable(self.db.cur, integer_iterable, column_name)

    def _regenerate_branch(self, perceptual_hash_id):
        log.info("reviewing existing branch")

        # grab everything in the branch

        (parent_id,) = self.db.execute(
            "SELECT parent_id FROM shape_vptree WHERE phash_id = ?;", (perceptual_hash_id,)
        ).fetchone()

        if parent_id is None:
            # this is the root node! we can't rebalance since there is no parent to spread across!

            self.db.execute("DELETE FROM shape_maintenance_branch_regen WHERE phash_id = ?;", (perceptual_hash_id,))

            return

        cte_table_name = "branch ( branch_phash_id )"
        initial_select = "SELECT ?"
        recursive_select = "SELECT phash_id FROM shape_vptree, branch ON parent_id = branch_phash_id"
        query_on_cte_table_name = (
            "SELECT branch_phash_id, phash FROM branch, shape_perceptual_hashes ON phash_id = branch_phash_id"
        )

        # use UNION (large memory, set), not UNION ALL (small memory, inifinite loop on damaged cyclic graph causing 200GB journal file and disk full error, jesus)  # noqa: E501
        query = "WITH RECURSIVE {} AS ( {} UNION {} ) {};".format(
            cte_table_name, initial_select, recursive_select, query_on_cte_table_name
        )

        unbalanced_nodes = self.db.execute(query, (perceptual_hash_id,)).fetchall()

        # removal of old branch, maintenance schedule, and orphan perceptual_hashes

        log.info(f"{len(unbalanced_nodes)} leaves found--now clearing out old branch")

        unbalanced_perceptual_hash_ids = {p_id for (p_id, p_h) in unbalanced_nodes}

        for p_id in unbalanced_perceptual_hash_ids:
            self.db.execute("DELETE FROM shape_vptree WHERE phash_id = ?;", ((p_id,)))

        self.clear_perceptual_hashes_from_vptree_node_cache(unbalanced_perceptual_hash_ids)

        for p_id in unbalanced_perceptual_hash_ids:
            self.db.execute("DELETE FROM shape_maintenance_branch_regen WHERE phash_id = ?;", ((p_id,)))

        with self._make_temporary_integer_table(
            unbalanced_perceptual_hash_ids, "phash_id"
        ) as temp_perceptual_hash_ids_table_name:
            useful_perceptual_hash_ids = self._STS(
                self.db.execute(
                    "SELECT phash_id FROM {} CROSS JOIN shape_perceptual_hash_map USING ( phash_id );".format(
                        temp_perceptual_hash_ids_table_name
                    )
                )
            )

        orphan_perceptual_hash_ids = unbalanced_perceptual_hash_ids.difference(useful_perceptual_hash_ids)

        for p_id in orphan_perceptual_hash_ids:
            self.db.execute("DELETE FROM shape_perceptual_hashes WHERE phash_id = ?;", ((p_id,)))

        useful_nodes = [row for row in unbalanced_nodes if row[0] in useful_perceptual_hash_ids]

        useful_population = len(useful_nodes)

        # now create the new branch, starting by choosing a new root and updating the parent's left/right reference to that  # noqa: E501

        if useful_population > 0:
            (new_perceptual_hash_id, new_perceptual_hash) = self.pop_best_root_node(useful_nodes)

        else:
            new_perceptual_hash_id = None
            new_perceptual_hash = None

        result = self.db.execute("SELECT inner_id FROM shape_vptree WHERE phash_id = ?;", (parent_id,)).fetchone()

        if result is None:
            # expected parent is not in the tree!
            # somehow some stuff got borked

            self.db.execute("DELETE FROM shape_maintenance_branch_regen;")

            msg = (
                "Your similar files search tree seemed to be damaged. Please regenerate it using '--clear-search-tree'!"
            )
            log.error(msg)
            print(msg)

            return

        (parent_inner_id,) = result

        if parent_inner_id == perceptual_hash_id:
            query = "UPDATE shape_vptree SET inner_id = ?, inner_population = ? WHERE phash_id = ?;"

        else:
            query = "UPDATE shape_vptree SET outer_id = ?, outer_population = ? WHERE phash_id = ?;"

        self.db.execute(query, (new_perceptual_hash_id, useful_population, parent_id))

        self.clear_perceptual_hashes_from_vptree_node_cache((parent_id,))

        if useful_population > 0:
            self.generate_branch(parent_id, new_perceptual_hash_id, new_perceptual_hash, useful_nodes)

    def maintain_tree(self):
        try:
            log.info("running similar files metadata maintenance")

            rebalance_perceptual_hash_ids = self._STL(
                self.db.execute("SELECT phash_id FROM shape_maintenance_branch_regen;")
            )

            num_to_do = len(rebalance_perceptual_hash_ids)

            while len(rebalance_perceptual_hash_ids) > 0:
                num_done = num_to_do - len(rebalance_perceptual_hash_ids)

                log.info(f"rebalancing similar file metadata - num_done: {num_done}, num_to_do: {num_to_do}")

                with self._make_temporary_integer_table(rebalance_perceptual_hash_ids, "phash_id") as temp_table_name:
                    # temp perceptual hashes to tree
                    result = self.db.execute(
                        "SELECT phash_id FROM {} CROSS JOIN shape_vptree USING ( phash_id ) ORDER BY inner_population + outer_population DESC;".format(  # noqa: E501
                            temp_table_name
                        )
                    ).fetchone()

                    if result is None:
                        self.db.execute("DELETE FROM shape_maintenance_branch_regen;")

                        return

                    else:
                        (biggest_perceptual_hash_id,) = result

                self._regenerate_branch(biggest_perceptual_hash_id)

                rebalance_perceptual_hash_ids = self._STL(
                    self.db.execute("SELECT phash_id FROM shape_maintenance_branch_regen;")
                )

        finally:
            log.info("done!")

    def search_perceptual_hashes(self, search_perceptual_hashes: Collection[bytes], max_hamming_distance: int) -> list:
        similar_hash_ids_and_distances = []

        if len(search_perceptual_hashes) == 0:
            return similar_hash_ids_and_distances

        if max_hamming_distance == 0:
            perceptual_hash_ids = set()

            for search_perceptual_hash in search_perceptual_hashes:
                perceptual_hash_id = self.db.get_phash_id(search_perceptual_hash)

                if perceptual_hash_id is not None:
                    perceptual_hash_ids.add(perceptual_hash_id)

            if len(perceptual_hash_ids) > 0:
                with self._make_temporary_integer_table(perceptual_hash_ids, "phash_id") as temp_table_name:
                    similar_hash_ids = self._STL(
                        self.db.execute(
                            f"SELECT hash_id FROM shape_perceptual_hash_map NATURAL JOIN {temp_table_name};"
                        )
                    )

                similar_hash_ids_and_distances.extend([(similar_hash_id, 0) for similar_hash_id in similar_hash_ids])

        else:
            search_radius = max_hamming_distance

            if self._root_node_perceptual_hash_id is None:
                top_node_result = self.db.execute(
                    "SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;"
                ).fetchone()

                if top_node_result is None:
                    return similar_hash_ids_and_distances

                (self._root_node_perceptual_hash_id,) = top_node_result

            similar_perceptual_hash_ids_to_distances = {}

            num_cycles = 0
            total_nodes_searched = 0

            for search_perceptual_hash in search_perceptual_hashes:
                # Fast path: precompute this query's distance to every indexed node in one go (exact same values
                # as calculate_distance). The tree walk below is unchanged; it just looks the value up.
                fast_distances = None
                fast_positions = None
                next_potentials = [self._root_node_perceptual_hash_id]
                if self._search_distance_index is not None:
                    fast_positions = self._search_distance_index.position_of
                    prefix = None
                    if self._search_scan_nodes is not None:
                        prefix = self._search_scan_prefix(search_radius)
                    if prefix is None:
                        fast_distances = self._search_distance_index.distances_for(search_perceptual_hash)
                    else:
                        fast_distances = self._search_distance_index.distance_array_for(search_perceptual_hash)
                        prefix_distances = None
                        if fast_distances is not None:
                            # Fast search scan: visit the whole prefix at once, in walk order (see scan_search_enabled).
                            prefix_positions, prefix_ids, prefix_next, prefix_unindexed = prefix
                            prefix_distances = fast_distances[prefix_positions]  # a copy
                            for k, node_radius, node_perceptual_hash in prefix_unindexed:
                                node_hamming_distance = calculate_distance(search_perceptual_hash, node_perceptual_hash)
                                if node_hamming_distance > node_radius + search_radius:
                                    prefix_distances = None  # the walk would not continue past this node: walk normally
                                    break
                                prefix_distances[k] = node_hamming_distance
                        if fast_distances is not None and prefix_distances is not None:
                            for k in np.flatnonzero(prefix_distances <= search_radius).tolist():
                                node_perceptual_hash_id = prefix_ids[k]
                                node_hamming_distance = int(prefix_distances[k])
                                current_distance = similar_perceptual_hash_ids_to_distances.get(node_perceptual_hash_id)
                                if current_distance is None:
                                    similar_perceptual_hash_ids_to_distances[node_perceptual_hash_id] = node_hamming_distance
                                else:
                                    similar_perceptual_hash_ids_to_distances[node_perceptual_hash_id] = min(
                                        node_hamming_distance, current_distance
                                    )
                            num_cycles += len(prefix_ids)
                            total_nodes_searched += len(prefix_ids)
                            next_potentials = [prefix_next]  # the last prefix node's inner child

                while len(next_potentials) > 0:
                    current_potentials = next_potentials
                    next_potentials = []

                    num_cycles += 1
                    total_nodes_searched += len(current_potentials)

                    # this is no longer an iterable inside the main node SELECT because it was causing crashes on linux!!  # noqa: E501
                    # after investigation, it seemed to be SQLite having a problem with part of Get64BitHammingDistance touching perceptual_hashes it presumably was still hanging on to  # noqa: E501
                    # the crash was in sqlite code, again presumably on subsequent fetch
                    # adding a fake delay in seemed to fix it also. guess it was some memory maintenance buffer/bytes thing  # noqa: E501
                    # anyway, we now just get the whole lot of results first and then work on the whole lot
                    # UPDATE: we moved to a cache finally, so the iteration danger is less worrying, but leaving the above up anyway  # noqa: E501

                    self._try_to_populate_perceptual_hash_to_vptree_node_cache(current_potentials)

                    for node_perceptual_hash_id in current_potentials:
                        if node_perceptual_hash_id not in self._perceptual_hash_id_to_vp_tree_node_cache:
                            # something crazy happened, probably a broken tree branch, move on
                            continue

                        (node_perceptual_hash, node_radius, inner_perceptual_hash_id, outer_perceptual_hash_id) = (
                            self._perceptual_hash_id_to_vp_tree_node_cache[node_perceptual_hash_id]
                        )

                        # first check the node itself--is it similar?

                        node_hamming_distance = None
                        if fast_distances is not None:
                            fast_position = fast_positions.get(node_perceptual_hash_id)
                            if fast_position is not None:
                                node_hamming_distance = int(fast_distances[fast_position])
                        if node_hamming_distance is None:  # original path (also the fallback)
                            node_hamming_distance = calculate_distance(search_perceptual_hash, node_perceptual_hash)

                        if node_hamming_distance <= search_radius:
                            if node_perceptual_hash_id in similar_perceptual_hash_ids_to_distances:
                                current_distance = similar_perceptual_hash_ids_to_distances[node_perceptual_hash_id]

                                similar_perceptual_hash_ids_to_distances[node_perceptual_hash_id] = min(
                                    node_hamming_distance, current_distance
                                )

                            else:
                                similar_perceptual_hash_ids_to_distances[node_perceptual_hash_id] = (
                                    node_hamming_distance
                                )

                        # now how about its children--where should we search next?

                        if node_radius is not None:
                            # we have two spheres--node and search--their centers separated by node_hamming_distance
                            # we want to search inside/outside the node_sphere if the search_sphere intersects with those spaces  # noqa: E501
                            # there are four possibles:
                            # (----N----)-(--S--)    intersects with outer only - distance between N and S > their radii
                            # (----N---(-)-S--)      intersects with both
                            # (----N-(--S-)-)        intersects with both
                            # (---(-N-S--)-)         intersects with inner only - distance between N and S + radius_S does not exceed radius_N  # noqa: E501

                            if inner_perceptual_hash_id is not None:
                                spheres_disjoint = node_hamming_distance > (node_radius + search_radius)

                                if not spheres_disjoint:  # i.e. they intersect at some point
                                    next_potentials.append(inner_perceptual_hash_id)

                            if outer_perceptual_hash_id is not None:
                                search_sphere_subset_of_node_sphere = (
                                    node_hamming_distance + search_radius
                                ) <= node_radius

                                if (
                                    not search_sphere_subset_of_node_sphere
                                ):  # i.e. search sphere intersects with non-node sphere space at some point
                                    next_potentials.append(outer_perceptual_hash_id)

            log.debug(f"Similar file search touched {total_nodes_searched} nodes over {num_cycles} cycles.")

            # so, now we have perceptual_hash_ids and distances. let's map that to actual files.
            # files can have multiple perceptual_hashes, and perceptual_hashes can refer to multiple files, so let's make sure we are setting the smallest distance we found  # noqa: E501

            similar_perceptual_hash_ids = list(similar_perceptual_hash_ids_to_distances.keys())

            with self._make_temporary_integer_table(similar_perceptual_hash_ids, "phash_id") as temp_table_name:
                # temp perceptual_hashes to hash map
                similar_perceptual_hash_ids_to_hash_ids = build_key_to_list_dict(
                    self.db.execute(
                        "SELECT phash_id, hash_id FROM {} CROSS JOIN shape_perceptual_hash_map USING ( phash_id );".format(  # noqa: E501
                            temp_table_name
                        )
                    )
                )

            similar_hash_ids_to_distances = {}

            for perceptual_hash_id, hash_ids in similar_perceptual_hash_ids_to_hash_ids.items():
                distance = similar_perceptual_hash_ids_to_distances[perceptual_hash_id]

                for hash_id in hash_ids:
                    if hash_id not in similar_hash_ids_to_distances:
                        similar_hash_ids_to_distances[hash_id] = distance

                    else:
                        current_distance = similar_hash_ids_to_distances[hash_id]

                        if distance < current_distance:
                            similar_hash_ids_to_distances[hash_id] = distance

            similar_hash_ids_and_distances.extend(similar_hash_ids_to_distances.items())

        similar_hash_ids_and_distances = dedupe_list(similar_hash_ids_and_distances)

        return similar_hash_ids_and_distances

    def _try_to_populate_perceptual_hash_to_vptree_node_cache(self, perceptual_hash_ids: Collection[int]):
        if len(self._perceptual_hash_id_to_vp_tree_node_cache) > 1000000:
            if not isinstance(perceptual_hash_ids, set):
                perceptual_hash_ids = set(perceptual_hash_ids)

            self._perceptual_hash_id_to_vp_tree_node_cache = {
                perceptual_hash_id: phash
                for (perceptual_hash_id, phash) in self._perceptual_hash_id_to_vp_tree_node_cache.items()
                if perceptual_hash_id in perceptual_hash_ids
            }

        uncached_perceptual_hash_ids = {
            perceptual_hash_id
            for perceptual_hash_id in perceptual_hash_ids
            if perceptual_hash_id not in self._perceptual_hash_id_to_vp_tree_node_cache
            and perceptual_hash_id not in self._non_vp_treed_perceptual_hash_ids
        }

        if len(uncached_perceptual_hash_ids) > 0:
            if len(uncached_perceptual_hash_ids) == 1:
                (uncached_perceptual_hash_id,) = uncached_perceptual_hash_ids

                rows = self.db.execute(
                    "SELECT phash_id, phash, radius, inner_id, outer_id FROM shape_perceptual_hashes CROSS JOIN shape_vptree USING ( phash_id ) WHERE phash_id = ?;",  # noqa: E501
                    (uncached_perceptual_hash_id,),
                ).fetchall()

            else:
                with self._make_temporary_integer_table(uncached_perceptual_hash_ids, "phash_id") as temp_table_name:
                    # temp perceptual_hash_ids to actual perceptual_hashes and tree info
                    rows = self.db.execute(
                        "SELECT phash_id, phash, radius, inner_id, outer_id FROM {} CROSS JOIN shape_perceptual_hashes USING ( phash_id ) CROSS JOIN shape_vptree USING ( phash_id );".format(  # noqa: E501
                            temp_table_name
                        )
                    ).fetchall()

            uncached_perceptual_hash_ids_to_vp_tree_nodes = {
                perceptual_hash_id: (phash, radius, inner_id, outer_id)
                for (perceptual_hash_id, phash, radius, inner_id, outer_id) in rows
            }

            if len(uncached_perceptual_hash_ids_to_vp_tree_nodes) < len(uncached_perceptual_hash_ids):
                for perceptual_hash_id in uncached_perceptual_hash_ids:
                    if perceptual_hash_id not in uncached_perceptual_hash_ids_to_vp_tree_nodes:
                        self._non_vp_treed_perceptual_hash_ids.add(perceptual_hash_id)

            self._perceptual_hash_id_to_vp_tree_node_cache.update(uncached_perceptual_hash_ids_to_vp_tree_nodes)

    def search_file(self, hash_id: int, max_hamming_distance: int) -> list:
        similar_hash_ids_and_distances = [(hash_id, 0)]

        # Videos don't have pixel hash ids. What do ?
        # pixel_hash_id = self._GetPixelHashId(hash_id)

        # if pixel_hash_id is not None:
        #     similar_hash_ids_and_distances.extend(self.SearchPixelHashes((pixel_hash_id,)))

        # IDENTICAL phashes. Not extremely similar, literally identical.
        if max_hamming_distance == 0:
            exact_match_hash_ids = self._STL(
                self.db.execute(
                    "SELECT hash_id FROM shape_perceptual_hash_map WHERE phash_id IN ( SELECT phash_id FROM shape_perceptual_hash_map WHERE hash_id = ? );",  # noqa: E501
                    (hash_id,),
                )
            )

            similar_hash_ids_and_distances.extend(
                [(exact_match_hash_id, 0) for exact_match_hash_id in exact_match_hash_ids]
            )

        else:
            # Note: This is different than Hydrus. In Hydrus one file can have multiple perceptual hashes for some reason.   # noqa: E501

            perceptual_hash_id = self.db.get_phash_id_from_hash_id(hash_id)
            assert perceptual_hash_id is not None

            perceptual_hashes = [self.db.get_phash(perceptual_hash_id)]
            assert perceptual_hashes is not None

            similar_hash_ids_and_distances.extend(
                self.search_perceptual_hashes(perceptual_hashes, max_hamming_distance)
            )

        similar_hash_ids_and_distances = dedupe_list(similar_hash_ids_and_distances)

        return similar_hash_ids_and_distances

    def maintenance_due(self) -> bool:
        """Whether there are unbalanced branches waiting to be rebuilt."""
        return self.db.execute("SELECT phash_id FROM shape_maintenance_branch_regen LIMIT 1;").fetchone() is not None

    def reset_search(self, hash_ids: list[int]):
        """Clear the search cache for the given hash ids."""
        for hash_id in hash_ids:
            self.db.execute(
                "UPDATE shape_search_cache SET searched_distance = NULL WHERE hash_id = :hash_id;",
                {"hash_id": hash_id},
            )
