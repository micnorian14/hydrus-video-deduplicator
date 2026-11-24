from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from rich import print

from ..__about__ import __version__
from ..dedup_util import print_user
from .vptree import VpTreeManager, chain_tree_rows, flush_pending_tree_writes, splice_out_tree_nodes


dedupedblog = logging.getLogger("db")
dedupedblog.setLevel(logging.INFO)

_db_dir: Path = Path()
_DB_FILE_NAME: str = "videohashes.sqlite"


class DedupeDbException(Exception):
    """Base class for DedupeDb exceptions."""


def does_db_exist() -> bool:
    """
    Check if the database exists.
    """
    db_path = get_db_file_path()
    try:
        _ = db_path.resolve(strict=True)
        return True
    except FileNotFoundError:
        return False


def create_db_dir() -> None:
    """
    Create the database folder if it doesn't already exist.
    """
    try:
        db_dir = get_db_file_path().parent
        os.makedirs(db_dir, exist_ok=False)
        # Exception before this log if directory already exists
        dedupedblog.info(f"Created DB dir {db_dir}")
    except OSError:
        pass


@dataclass
class DatabaseStats:
    num_videos: int
    file_size: int  # in bytes


def get_db_stats(db: DedupeDb) -> DatabaseStats:
    """Get some database stats."""
    num_videos = db.get_num_phashed_files()
    file_size = os.path.getsize(get_db_file_path())
    return DatabaseStats(num_videos, file_size)


# --backup: how many backups to keep, and the backup file names (e.g. videohashes.backup-20261121-013000.sqlite).
# Only files matching exactly this name pattern are ever deleted by the cleanup of old backups.
NUM_BACKUPS_TO_KEEP = 3


def backup_file_pattern(db_name: str | None = None) -> re.Pattern:
    """Regex for backup file names of db_name (default videohashes.sqlite)."""
    stem, ext = os.path.splitext(db_name or get_db_name())
    return re.compile(re.escape(stem) + r"\.backup-\d{8}-\d{6}" + re.escape(ext) + "$")


def backup_db(
    db_path: Path | None = None, keep: int = NUM_BACKUPS_TO_KEEP, *, removed: list | None = None
) -> Path | None:
    """
    Copy the database next to itself as <name>.backup-YYYYMMDD-HHMMSS.sqlite (--backup).

    Uses SQLite's online backup API, so the copy is consistent even if the database is open elsewhere. The copy is
    written to a temporary name first and only renamed once complete, so a failed or interrupted backup never
    counts as a backup. Afterwards only the newest `keep` backups are kept; older ones matching the backup name
    pattern are deleted (no other file is touched). Returns the backup path, or None if there is no database yet.
    If `removed` is a list, one (old backup path, OSError or None) tuple is appended per old backup it tried to delete.
    """
    db_path = Path(db_path) if db_path is not None else get_db_file_path()
    if not db_path.is_file():
        return None
    stem, ext = os.path.splitext(db_path.name)
    while True:
        backup_path = db_path.with_name(f"{stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}{ext}")
        if not backup_path.exists():
            break
        time.sleep(1)  # one backup per second at most, so names never collide
    partial_path = backup_path.with_name(backup_path.name + ".partial")

    src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(partial_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    except BaseException:
        src.close()
        try:
            os.remove(partial_path)
        except OSError:
            pass
        raise
    src.close()
    os.replace(partial_path, backup_path)

    pattern = backup_file_pattern(db_path.name)
    backups = sorted(path for path in db_path.parent.iterdir() if path.is_file() and pattern.match(path.name))
    for old in backups[: max(0, len(backups) - keep)]:
        try:
            os.remove(old)
            dedupedblog.info(f"Deleted old backup {old}")
            if removed is not None:
                removed.append((old, None))
        except OSError as exc:
            dedupedblog.warning(f"Could not delete old backup {old}: {exc}")
            if removed is not None:
                removed.append((old, exc))
    return backup_path


@dataclass
class PruneResult:
    videos: int  # distinct deleted videos removed (files table and/or phashed file queue)
    files: int  # rows removed from the files table
    queued: int  # rows removed from the phashed file queue
    tree_nodes: int  # perceptual hashes removed from the search tree
    perceptual_hashes: int  # perceptual hashes removed because no remaining file uses them


def normalize_file_hash(file_hash) -> str:
    """Lowercase hex string for a file hash stored as text or bytes."""
    if isinstance(file_hash, (bytes, bytearray, memoryview)):
        return bytes(file_hash).hex()
    return str(file_hash).strip().lower()


def create_db():
    """
    Create the database files.
    """
    if not get_db_dir().exists():
        create_db_dir()
    db = DedupeDb(get_db_dir(), get_db_name())
    db.init_connection()
    db.create_tables()
    db.commit()
    db.close()


def set_db_dir(dir: Path):
    """Set the directory for the database."""
    global _db_dir
    _db_dir = dir


def get_db_dir():
    """Get the directory of the database."""
    return _db_dir


def get_db_name() -> str:
    """Get the db file name, e.g. videohashes.sqlite."""
    return _DB_FILE_NAME


def get_db_file_path() -> Path:
    """
    Get database file path.

    Return the database file path.
    """
    return get_db_dir() / get_db_name()


class DedupeDb:
    def __init__(self, db_dir: Path, db_name: str):
        self.db_dir = db_dir
        self.db_name = db_name

        self.conn = None
        self.cur = None

    """
    Main functions
    """

    def execute(self, query, *query_args) -> sqlite3.Cursor:
        return self.cur.execute(query, *query_args)

    def set_cursor(self, cur: sqlite3.Cursor):
        self.cur = cur

    def init_connection(self):
        db_path = self.db_dir / self.db_name
        self.conn = sqlite3.connect(db_path)
        cur = self.conn.cursor()
        self.set_cursor(cur)

    def commit(self):
        flush_pending_tree_writes(self)  # deferred fast tree building counts (no-op otherwise)
        self.conn.commit()

    def begin_transaction(self):
        self.execute("BEGIN TRANSACTION")

    def close(self):
        self.conn.close()

    """
    Tables
    """

    def create_tables(self):
        # version table
        self.execute("CREATE TABLE IF NOT EXISTS version (version TEXT)")
        self.execute("INSERT INTO version (version) VALUES (:version)", {"version": __version__})

        # files table
        self.execute("CREATE TABLE IF NOT EXISTS files ( hash_id INTEGER PRIMARY KEY, file_hash BLOB_BYTES UNIQUE )")

        # phash tables
        self.execute(
            "CREATE TABLE IF NOT EXISTS shape_perceptual_hashes ( phash_id INTEGER PRIMARY KEY, phash BLOB_BYTES UNIQUE )"  # noqa: E501
        )
        self.execute(
            "CREATE TABLE IF NOT EXISTS shape_perceptual_hash_map ( phash_id INTEGER, hash_id INTEGER, PRIMARY KEY ( phash_id, hash_id ) )"  # noqa: E501
        )

        # vptree tables
        self.execute(
            "CREATE TABLE IF NOT EXISTS shape_vptree ( phash_id INTEGER PRIMARY KEY, parent_id INTEGER, radius INTEGER, inner_id INTEGER, inner_population INTEGER, outer_id INTEGER, outer_population INTEGER )"  # noqa: E501
        )
        # fmt: off
        self.execute(
            "CREATE TABLE IF NOT EXISTS shape_maintenance_branch_regen ( phash_id INTEGER PRIMARY KEY )"
        )
        # fmt: on
        self.execute(
            "CREATE TABLE IF NOT EXISTS shape_search_cache ( hash_id INTEGER PRIMARY KEY, searched_distance INTEGER )"
        )

        # vptree insert queue. this is the list of files and their phashes that need to be inserted into the vptree.
        # when entries are added to this queue they don't exist at all in the other tables. they don't have a hash_id
        # or phash_id yet, unless those already exist from other files.
        # this is just a table to store the phashes until they are properly inserted into the vptree, since inserting
        # can take a while.
        self.execute(
            "CREATE TABLE IF NOT EXISTS phashed_file_queue ( file_hash BLOB_BYTES NOT NULL UNIQUE, phash BLOB_BYTES NOT NULL, PRIMARY KEY ( file_hash, phash ) )"  # noqa: E501
        )

    """
    Utility
    """

    def clear_search_tree(self):
        """Clear the search tree. The search cache will also be cleared. Does not clear the perceptual hash map."""
        # Note: Need a separate cursor here since we're running queries in the loop that overwrite this cursor.
        cur = self.conn.cursor()
        cur.execute("SELECT phash_id, hash_id FROM shape_perceptual_hash_map")

        # Move the files back into the queue so that the tree can be rebuilt.
        for phash_id, hash_id in cur:
            phash_result = self.execute(
                "SELECT phash FROM shape_perceptual_hashes WHERE phash_id = :phash_id", {"phash_id": phash_id}
            ).fetchone()
            if not phash_result:
                # This should not happen. Perceptual hashes should always be in the database if there is a phash_id,
                # otherwise we have no idea what perceptual hash it is.
                print_user(
                    f"ERROR clearing search tree while to get perceptual_hash from phash_id {phash_id}. perceptual_hash not found. Your DB may be corrupt."  # noqa: E501
                )
                continue
            perceptual_hash = phash_result[0]

            file_hash_result = self.execute(
                "SELECT file_hash FROM files WHERE hash_id = :hash_id", {"hash_id": hash_id}
            ).fetchone()
            if not file_hash_result:
                # This should not happen. File hashes should always be in the database if there is a hash_id,
                # otherwise we have no idea what file it is in Hydrus.
                print_user(
                    f"ERROR clearing search tree while to get file_hash from hash_id {hash_id}. file_hash not found. Your DB may be corrupt."  # noqa: E501
                )
                continue
            file_hash = file_hash_result[0]

            self.add_to_phashed_files_queue(file_hash, perceptual_hash)

        self.execute("DELETE FROM shape_vptree")
        self.execute("DELETE FROM shape_search_cache")
        self.execute("DELETE FROM shape_maintenance_branch_regen")

    def relink_search_tree(self) -> dict[str, int]:
        """
        Repair a damaged search tree without re-inserting files one at a time.

        Every perceptual hash used by a file is linked into one valid tree (see vptree.chain_tree_rows), so every
        file is reachable by the duplicate search again. Then the search cache is reset so the next runs search
        every file again. Nothing is re-hashed and the phashed file queue is left alone. Files whose perceptual
        hash or file row is missing are skipped, like clear_search_tree() does (nothing is deleted for them).

        The end state is equivalent to clear_search_tree() followed by a full "Building search tree" phase: the
        same files are in the tree and in the search cache (with searched_distance NULL), so the same duplicate
        pairs are found. Only the tree layout differs. Run it inside a transaction so it's all-or-nothing.
        """
        broken_phash = self.execute(
            "SELECT COUNT(*) FROM shape_perceptual_hash_map WHERE phash_id NOT IN ( SELECT phash_id FROM shape_perceptual_hashes );"  # noqa: E501
        ).fetchone()[0]
        broken_file = self.execute(
            "SELECT COUNT(*) FROM shape_perceptual_hash_map WHERE hash_id NOT IN ( SELECT hash_id FROM files );"
        ).fetchone()[0]

        # The files clear_search_tree() would requeue (valid perceptual hash AND file row), and their hashes.
        valid_map = (
            "shape_perceptual_hash_map JOIN shape_perceptual_hashes USING ( phash_id ) JOIN files USING ( hash_id )"
        )
        phash_ids = [
            phash_id
            for (phash_id,) in self.execute(f"SELECT DISTINCT phash_id FROM {valid_map} ORDER BY phash_id;").fetchall()
        ]

        self.execute("DELETE FROM shape_vptree;")
        self.cur.executemany(
            "INSERT INTO shape_vptree ( phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population ) VALUES ( ?, ?, ?, ?, ?, ?, ? );",  # noqa: E501
            chain_tree_rows(phash_ids),
        )
        self.execute("DELETE FROM shape_maintenance_branch_regen;")

        # Same search cache as after clear_search_tree() + rebuilding: one row per valid file, not yet searched.
        self.execute("DELETE FROM shape_search_cache;")
        self.execute(
            "INSERT INTO shape_search_cache ( hash_id, searched_distance ) "
            f"SELECT DISTINCT hash_id, NULL FROM {valid_map};"
        )
        files_to_search = self.execute("SELECT COUNT(*) FROM shape_search_cache;").fetchone()[0]

        return {
            "tree_nodes": len(phash_ids),
            "files_to_search": files_to_search,
            "skipped_missing_phash": broken_phash,
            "skipped_missing_file": broken_file,
        }

    # Files whose perceptual hash row is missing (an old upgrade could delete hashes another row still used).
    ORPHANED_FILES_QUERY = (
        "SELECT DISTINCT hash_id FROM shape_perceptual_hash_map "
        "WHERE phash_id NOT IN ( SELECT phash_id FROM shape_perceptual_hashes )"
    )

    def count_files(self) -> int:
        return self.execute("SELECT COUNT(*) FROM files;").fetchone()[0]

    def count_orphaned_files(self) -> int:
        """Number of files whose perceptual hash is missing. Read only; one query (about 0.2 s on 400k files)."""
        return self.execute(f"SELECT COUNT(*) FROM ( {self.ORPHANED_FILES_QUERY} );").fetchone()[0]

    def check_search_tree(self) -> dict[str, int]:
        """
        Read-only check for search tree damage that --relink-search-tree repairs (issue #80). About 0.5 s on 400k.

        Counts, for the tree the search walks (from the root it uses, `parent_id IS NULL`):
          extra_roots     more than one root (only one is searched)
          no_root         1 if the tree has nodes but no root
          broken_links    inner/outer links to nodes that don't exist
          shared          nodes linked from more than one parent (or from a loop)
          unreachable     nodes the search can't reach from the root (detached branches, loops)
          wrong_parent    nodes whose parent_id isn't the node that links to them
          no_hash         tree nodes without a perceptual hash row
          not_in_tree     perceptual hashes used by a file but missing from the tree, not counting files that are
                          still waiting in the phashed file queue (normal until "Building search tree" finishes)
        "problems" is their sum. Pending branch maintenance (shape_maintenance_branch_regen) is normal and ignored.
        """
        import numpy as np

        rows = self.execute(
            "SELECT phash_id, IFNULL( parent_id, -1 ), IFNULL( inner_id, -1 ), IFNULL( outer_id, -1 ) "
            "FROM shape_vptree ORDER BY phash_id;"
        ).fetchall()
        tree = np.array(rows, dtype=np.int64).reshape(-1, 4)
        del rows
        ids, parents, inners, outers = tree[:, 0], tree[:, 1], tree[:, 2], tree[:, 3]
        n = len(ids)
        result = dict.fromkeys(("extra_roots", "no_root", "broken_links", "shared", "unreachable", "wrong_parent"), 0)
        if n > 0:
            # Every link (parent position -> child id), and whether the child exists.
            link_from = np.concatenate([np.arange(n), np.arange(n)])
            link_to = np.concatenate([inners, outers])
            used = link_to >= 0
            link_from, link_to = link_from[used], link_to[used]
            child = np.minimum(np.searchsorted(ids, link_to), n - 1)
            exists = ids[child] == link_to
            result["broken_links"] = int(np.count_nonzero(~exists))
            link_from, child = link_from[exists], child[exists]

            incoming = np.bincount(child, minlength=n)
            result["shared"] = int(np.maximum(incoming - 1, 0).sum())
            result["wrong_parent"] = int(np.count_nonzero(ids[link_from] != parents[child]))

            roots = self.execute("SELECT COUNT(*) FROM shape_vptree WHERE parent_id IS NULL;").fetchone()[0]
            result["extra_roots"] = max(0, roots - 1)
            root_row = self.execute("SELECT phash_id FROM shape_vptree WHERE parent_id IS NULL;").fetchone()
            if root_row is None:
                result["no_root"] = 1
                result["unreachable"] = n
            else:
                # Follow one parent link per node up to the root (pointer jumping). Nodes that never reach the
                # root are unreachable: detached branches or loops.
                root = int(np.searchsorted(ids, root_row[0]))
                up = np.arange(n)
                up[child] = link_from
                up[root] = root
                for _ in range(64):  # 2**64 steps is far more than any tree depth
                    next_up = up[up]
                    if np.array_equal(next_up, up):
                        break
                    up = next_up
                result["unreachable"] = int(np.count_nonzero(up != root))

        result["no_hash"] = self.execute(
            "SELECT COUNT(*) FROM shape_vptree WHERE phash_id NOT IN ( SELECT phash_id FROM shape_perceptual_hashes );"
        ).fetchone()[0]
        result["not_in_tree"] = self.execute(
            "SELECT COUNT( DISTINCT phash_id ) FROM shape_perceptual_hash_map "
            "JOIN shape_perceptual_hashes USING ( phash_id ) JOIN files USING ( hash_id ) "
            "WHERE phash_id NOT IN ( SELECT phash_id FROM shape_vptree ) "
            "AND file_hash NOT IN ( SELECT file_hash FROM phashed_file_queue );"
        ).fetchone()[0]
        result["problems"] = sum(result.values())
        result["nodes"] = n
        return result

    def remove_orphaned_files(self) -> int:
        """
        Remove files whose perceptual hash is missing (--prune-orphans). Run it inside a transaction.

        Deletes their broken map rows and search cache rows, and their files rows unless the file also has a valid
        perceptual hash or is waiting in the phashed file queue. The next run then hashes them again like new files.
        Perceptual hashes and the search tree are not touched. Returns the number of files removed.
        """
        hash_ids = [(hash_id,) for (hash_id,) in self.execute(f"{self.ORPHANED_FILES_QUERY};").fetchall()]
        if not hash_ids:
            return 0
        self.cur.executemany(
            "DELETE FROM shape_perceptual_hash_map WHERE hash_id = ? "
            "AND phash_id NOT IN ( SELECT phash_id FROM shape_perceptual_hashes );",
            hash_ids,
        )
        self.cur.executemany(
            "DELETE FROM files WHERE hash_id = ? "
            "AND hash_id NOT IN ( SELECT hash_id FROM shape_perceptual_hash_map ) "
            "AND file_hash NOT IN ( SELECT file_hash FROM phashed_file_queue );",
            hash_ids,
        )
        self.cur.executemany(
            "DELETE FROM shape_search_cache WHERE hash_id = ? "
            "AND hash_id NOT IN ( SELECT hash_id FROM shape_perceptual_hash_map );",
            hash_ids,
        )
        return len(hash_ids)

    def prune_files_not_in(self, existing_file_hashes) -> PruneResult:
        """
        Remove every file that is not in existing_file_hashes (--prune-deleted-files). Run it inside a transaction.

        Each removed file loses its files row, perceptual hash map rows, search cache row and phashed file queue row.
        A perceptual hash that no remaining file uses is removed from shape_perceptual_hashes, the search tree and
        the branch maintenance list; the tree is kept valid with vptree.splice_out_tree_nodes. Nothing else changes:
        the remaining files keep their search cache, so already searched files are not searched again.
        """
        existing = {normalize_file_hash(file_hash) for file_hash in existing_file_hashes}

        file_rows = self.execute("SELECT hash_id, file_hash FROM files;").fetchall()
        gone_files = [
            (hash_id, file_hash) for hash_id, file_hash in file_rows if normalize_file_hash(file_hash) not in existing
        ]
        queue_rows = self.execute("SELECT file_hash FROM phashed_file_queue;").fetchall()
        gone_queue = [file_hash for (file_hash,) in queue_rows if normalize_file_hash(file_hash) not in existing]
        del file_rows, queue_rows
        videos = len(
            {normalize_file_hash(file_hash) for _, file_hash in gone_files}
            | {normalize_file_hash(file_hash) for file_hash in gone_queue}
        )
        tree_nodes_removed, perceptual_hashes_removed = self._remove_files(gone_files, gone_queue)
        return PruneResult(
            videos=videos,
            files=len(gone_files),
            queued=len(gone_queue),
            tree_nodes=tree_nodes_removed,
            perceptual_hashes=perceptual_hashes_removed,
        )

    def _remove_files(self, gone_files: list, gone_queue: list) -> tuple[int, int]:
        """
        Remove files (hash_id, file_hash) and queue entries (file_hash) like prune_files_not_in(). Run it inside a
        transaction. Returns (search tree nodes removed, perceptual hashes removed).
        """
        self.execute("CREATE TEMP TABLE IF NOT EXISTS hvd_prune_hash_ids ( hash_id INTEGER PRIMARY KEY );")
        try:
            self.execute("DELETE FROM temp.hvd_prune_hash_ids;")
            self.cur.executemany(
                "INSERT INTO temp.hvd_prune_hash_ids ( hash_id ) VALUES ( ? );",
                [(hash_id,) for hash_id, _ in gone_files],
            )
            in_pruned = "hash_id IN ( SELECT hash_id FROM temp.hvd_prune_hash_ids )"
            affected_phash_ids = {
                phash_id
                for (phash_id,) in self.execute(
                    f"SELECT DISTINCT phash_id FROM shape_perceptual_hash_map WHERE {in_pruned};"
                )
            }
            self.execute(f"DELETE FROM shape_perceptual_hash_map WHERE {in_pruned};")
            self.execute(f"DELETE FROM shape_search_cache WHERE {in_pruned};")
            self.execute(f"DELETE FROM files WHERE {in_pruned};")
        finally:
            self.execute("DROP TABLE IF EXISTS temp.hvd_prune_hash_ids;")
        self.cur.executemany(
            "DELETE FROM phashed_file_queue WHERE file_hash = ?;", [(file_hash,) for file_hash in gone_queue]
        )

        still_used = set()
        if affected_phash_ids:
            still_used = {
                phash_id
                for (phash_id,) in self.execute("SELECT DISTINCT phash_id FROM shape_perceptual_hash_map;")
                if phash_id in affected_phash_ids
            }
        orphan_phash_ids = affected_phash_ids - still_used

        tree_nodes_removed = 0
        if orphan_phash_ids:
            rows = self.execute(
                "SELECT phash_id, parent_id, radius, inner_id, inner_population, outer_id, outer_population FROM shape_vptree;"  # noqa: E501
            ).fetchall()
            before = {row[0]: row[1:] for row in rows}
            del rows
            nodes = {phash_id: list(values) for phash_id, values in before.items()}
            tree_nodes_removed = splice_out_tree_nodes(nodes, orphan_phash_ids)
            self.cur.executemany(
                "DELETE FROM shape_vptree WHERE phash_id = ?;",
                [(phash_id,) for phash_id in before if phash_id not in nodes],
            )
            self.cur.executemany(
                "UPDATE shape_vptree SET parent_id = ?, radius = ?, inner_id = ?, inner_population = ?, outer_id = ?, outer_population = ? WHERE phash_id = ?;",  # noqa: E501
                [(*node, phash_id) for phash_id, node in nodes.items() if tuple(node) != before[phash_id]],
            )
            del before, nodes
            self.cur.executemany(
                "DELETE FROM shape_perceptual_hashes WHERE phash_id = ?;", [(p,) for p in orphan_phash_ids]
            )
            self.cur.executemany(
                "DELETE FROM shape_maintenance_branch_regen WHERE phash_id = ?;", [(p,) for p in orphan_phash_ids]
            )

        return tree_nodes_removed, len(orphan_phash_ids)

    def get_hashed_frame_counts(self, min_frames: int = 0) -> list[tuple[str, int]]:
        """
        (file hash, number of hashed frames) for every hashed video with at least min_frames hashed frames, including
        the phashed file queue (--rehash-overhashed). Read only.
        """
        min_bytes = max(0, int(min_frames)) * 32  # each hashed frame is 32 bytes
        rows = self.execute(
            "SELECT f.file_hash, LENGTH(p.phash) / 32 FROM files f "
            "JOIN shape_perceptual_hash_map m ON m.hash_id = f.hash_id "
            "JOIN shape_perceptual_hashes p ON p.phash_id = m.phash_id "
            "WHERE LENGTH(p.phash) >= :min_bytes "
            "UNION ALL "
            "SELECT file_hash, LENGTH(phash) / 32 FROM phashed_file_queue WHERE LENGTH(phash) >= :min_bytes;",
            {"min_bytes": min_bytes},
        ).fetchall()
        counts: dict[str, int] = {}
        for file_hash, frames in rows:
            file_hash = normalize_file_hash(file_hash)
            counts[file_hash] = max(frames, counts.get(file_hash, 0))
        return list(counts.items())

    def get_stored_phash(self, file_hash: str) -> bytes | None:
        """The perceptual hash stored for a file (the queued one if it's waiting in the queue), or None. Read only."""
        row = self.execute(
            "SELECT phash FROM phashed_file_queue WHERE file_hash = :file_hash;", {"file_hash": file_hash}
        ).fetchone()
        if row is None:
            row = self.execute(
                "SELECT p.phash FROM files f "
                "JOIN shape_perceptual_hash_map m ON m.hash_id = f.hash_id "
                "JOIN shape_perceptual_hashes p ON p.phash_id = m.phash_id "
                "WHERE f.file_hash = :file_hash;",
                {"file_hash": file_hash},
            ).fetchone()
        return None if row is None else bytes(row[0])

    def replace_perceptual_hashes(self, new_phashes: dict[str, bytes]) -> PruneResult:
        """
        Give files a new perceptual hash (--rehash-overhashed). Run it inside a transaction.

        Each file is removed exactly like --prune-deleted-files removes a deleted video (its old perceptual hash
        leaves the search tree if no other file uses it), then queued with its new perceptual hash, so the next
        "Building search tree" step adds it back like a newly hashed video. Its search progress starts over.
        """
        wanted = {normalize_file_hash(file_hash): phash for file_hash, phash in new_phashes.items()}
        if not wanted:
            return PruneResult(videos=0, files=0, queued=0, tree_nodes=0, perceptual_hashes=0)
        file_rows = self.execute("SELECT hash_id, file_hash FROM files;").fetchall()
        gone_files = [
            (hash_id, file_hash) for hash_id, file_hash in file_rows if normalize_file_hash(file_hash) in wanted
        ]
        del file_rows
        # Queue entries are simply replaced below, so only the files table rows are removed here.
        tree_nodes_removed, perceptual_hashes_removed = self._remove_files(gone_files, [])
        stored_names = {normalize_file_hash(file_hash): file_hash for _, file_hash in gone_files}
        for file_hash, phash in wanted.items():
            self.add_to_phashed_files_queue(stored_names.get(file_hash, file_hash), phash)
        return PruneResult(
            videos=len(wanted),
            files=len(gone_files),
            queued=len(wanted),
            tree_nodes=tree_nodes_removed,
            perceptual_hashes=perceptual_hashes_removed,
        )

    def clear_search_cache(self):
        """Clear the search cache for all files."""
        tree = VpTreeManager(self)
        result = self.execute("SELECT hash_id FROM shape_search_cache").fetchall()
        if result:
            hash_ids = [hash_id[0] for hash_id in result]
            tree.reset_search(hash_ids)

    def add_file(self, file_hash: str):
        """Add a file to the db. If it already exists, do nothing."""
        self.execute("INSERT OR IGNORE INTO files ( file_hash ) VALUES ( :file_hash )", {"file_hash": file_hash})

    def add_perceptual_hash(self, perceptual_hash: bytes) -> int:
        """
        Add a perceptual hash to the db.
        If it already exists, do nothing.

        Returns the phash_id of the perceptual hash.
        """
        result = self.execute(
            "SELECT phash_id FROM shape_perceptual_hashes WHERE phash = :phash;", {"phash": perceptual_hash}
        ).fetchone()

        if result is None:
            self.execute(
                "INSERT INTO shape_perceptual_hashes ( phash ) VALUES ( :phash )",
                {"phash": perceptual_hash},
            )
            result = self.execute(
                "SELECT phash_id FROM shape_perceptual_hashes WHERE phash = :phash;", {"phash": perceptual_hash}
            ).fetchone()
            result = result[0]
            assert isinstance(result, int)
        else:
            result = result[0]
        # TODO: Double check that the return value here is actually there if the result is not None.
        # Remove the assert below if it is.
        assert isinstance(result, int)
        return result

    def add_to_phashed_files_queue(self, file_hash: str, perceptual_hash: bytes):
        """
        Add a file and its corresponding perceptual hash to the queue to be inserted into the vptree.

        We keep the queue of files to be inserted in the vptree in a separate table to avoid any potential issues
        with assumptions of what needs to exist when/where for vptree operations.

        If the file hash is already in the queue, it will be replaced with the new perceptual hash.
        """
        self.execute(
            "REPLACE INTO phashed_file_queue ( file_hash, phash ) VALUES ( :file_hash, :phash )",
            {"file_hash": file_hash, "phash": perceptual_hash},
        )

    def associate_file_with_perceptual_hash(self, file_hash: str, perceptual_hash: bytes):
        """
        Associate a file with a perceptual hash in the database.
        This will insert the file into the VpTree.
        If the file already has a perceptual hash, it will be overwritten.

        Note:
        Perceptual hashes are not unique for each file.
        Files can have identical perceptual hashes.
        This is not even that rare, e.g. a video that is all the same color.
        """
        hash_id = self.get_hash_id(file_hash)

        perceptual_hash_id = self.get_phash_id(perceptual_hash)
        assert perceptual_hash_id is not None

        tree = VpTreeManager(self)
        tree.add_leaf(perceptual_hash_id, perceptual_hash)

        already_exists = self.execute(
            "SELECT hash_id FROM shape_perceptual_hash_map WHERE hash_id = :hash_id", {"hash_id": hash_id}
        ).fetchone()

        if already_exists:
            self.execute("DELETE FROM shape_perceptual_hash_map WHERE hash_id = :hash_id", {"hash_id": hash_id})

        res = self.execute(
            "INSERT INTO shape_perceptual_hash_map ( phash_id, hash_id ) VALUES ( :phash_id, :hash_id )",
            {"phash_id": perceptual_hash_id, "hash_id": hash_id},
        )

        # NOTE: We must fetchone here so that the rowcount is updated.
        res.fetchone()
        if res.rowcount > 0:
            self.execute(
                "REPLACE INTO shape_search_cache ( hash_id, searched_distance ) VALUES ( :hash_id, :searched_distance );",  # noqa: E501
                {"hash_id": hash_id, "searched_distance": None},
            )

    def get_version(self) -> str:
        if self.does_table_exist("version"):
            (version,) = self.execute("SELECT version FROM version;").fetchone()
        else:
            # Old versions of the database did not have a version table. We will assume it's the most recent version
            # that had no version table.
            version = "0.6.0"
        return version

    def set_version(self, version: str):
        self.execute("UPDATE version SET version = :version", {"version": version})

    def does_table_exist(self, table: str) -> bool:
        # pls no injection. named placeholders don't work for tables.
        res = self.execute(f"SELECT * FROM pragma_table_list WHERE name='{table}'")
        return bool(res.fetchall())

    def get_phash_id(self, perceptual_hash: bytes) -> str | None:
        """Get the perceptual hash id from the phash, or None if not found."""
        result = self.execute(
            "SELECT phash_id FROM shape_perceptual_hashes WHERE phash = :phash", {"phash": perceptual_hash}
        ).fetchone()
        perceptual_hash_id = None
        if result is not None:
            (perceptual_hash_id,) = result
        return perceptual_hash_id

    def get_phash_id_from_hash_id(self, hash_id: str) -> str | None:
        """Get the phash id from the hash_id, or None if not found."""
        result = self.execute(
            "SELECT phash_id FROM shape_perceptual_hash_map WHERE hash_id = :hash_id", {"hash_id": hash_id}
        ).fetchone()
        perceptual_hash_id = None
        if result is not None:
            (perceptual_hash_id,) = result
        return perceptual_hash_id

    def get_hash_id(self, file_hash: str) -> str | None:
        """Get the hash id from the file hash, or None if not found."""
        result = self.execute(
            "SELECT hash_id FROM files WHERE file_hash = :file_hash", {"file_hash": file_hash}
        ).fetchone()
        hash_id = None
        if result is not None:
            (hash_id,) = result
        return hash_id

    def get_phash(self, phash_id: str) -> bytes | None:
        """Get the perceptual hash from its phash_id, or None if not found."""
        result = self.execute(
            "SELECT phash FROM shape_perceptual_hashes WHERE phash_id = :phash_id", {"phash_id": phash_id}
        ).fetchone()
        phash = None
        if result is not None:
            (phash,) = result
        return phash

    def get_file_hash(self, hash_id: str) -> str | None:
        """Get the file hash from its hash_id, or None if not found."""
        result = self.execute("SELECT file_hash FROM files WHERE hash_id = :hash_id", {"hash_id": hash_id}).fetchone()
        file_hash = None
        if result is not None:
            (file_hash,) = result
        return file_hash

    def get_phashed_files(self) -> list[str]:
        """Get the file hashes of all files that are phashed. This includes the files in the phashed_file_queue."""
        all_phashed_files_query = (
            "SELECT file_hash FROM files "
            "WHERE hash_id IN (SELECT hash_id FROM shape_perceptual_hash_map) "
            "UNION "
            "SELECT file_hash FROM phashed_file_queue"
        )
        all_phashed_files = self.execute(all_phashed_files_query)
        all_phashed_files = [row[0] for row in all_phashed_files]
        return all_phashed_files

    def count_queued_files(self) -> int:
        """Number of hashed files still waiting in the phashed_file_queue to be added to the search tree."""
        return self.execute("SELECT COUNT(*) FROM phashed_file_queue;").fetchone()[0]

    def get_num_phashed_files(self) -> int:
        """Get total number of all files that are phashed.  This includes the files in the phashed_file_queue."""
        query = """
            SELECT COUNT(*) FROM (
                SELECT file_hash FROM files
                WHERE hash_id IN (SELECT hash_id FROM shape_perceptual_hash_map)
                UNION
                SELECT file_hash FROM phashed_file_queue
            )
        """
        cursor = self.execute(query)
        row = cursor.fetchone()
        return row[0] if row else 0

    """
    Misc
    """

    def does_need_upgrade(self) -> bool:
        db_version = SemanticVersion(self.get_version())
        dedupe_version = SemanticVersion(__version__)
        return db_version < dedupe_version

    def vacuum(self):
        """
        Vacuum the db.

        Note: Can't vacuum within a transaction.
        """
        self.execute("VACUUM")

    def upgrade_db(self) -> bool:
        """
        Upgrade the db.

        Returns true if the DB needed to be upgraded and was upgraded.
        """

        # WARNING:
        # If any functions change then upgrades may not work! We need to be careful about updating them and what we call. # noqa: E501
        # An upgrade cutoff at some point to prevent bitrot is a good idea, which is what Hydrus does.

        def print_upgrade(version: str, new_version: str):
            print(f"Upgrading db from {version} to version {new_version}")

        version = self.get_version()
        if SemanticVersion(__version__) < SemanticVersion(version):
            raise DedupeDbException(
                f"""
Database version {version} is newer than the installed hydrusvideodeduplicator version {__version__}.\
\nPlease upgrade and try again. \
\nSee documentation for how to upgrade: https://github.com/hydrusvideodeduplicator/hydrus-video-deduplicator/blob/main/docs/faq.md#how-to-update
                """
            )

        if not self.does_need_upgrade():
            return False

        if SemanticVersion(version) < SemanticVersion("0.7.0"):
            print_upgrade(version, "0.7.0")

            # Create version table
            self.execute("CREATE TABLE IF NOT EXISTS version (version TEXT)")
            self.execute("INSERT INTO version (version) VALUES (:version)", {"version": "0.6.0"})

            # Create the vptree tables
            self.execute(
                "CREATE TABLE IF NOT EXISTS files ( hash_id INTEGER PRIMARY KEY, file_hash BLOB_BYTES UNIQUE )"
            )
            self.execute(
                "CREATE TABLE IF NOT EXISTS shape_perceptual_hashes ( phash_id INTEGER PRIMARY KEY, phash BLOB_BYTES UNIQUE )"  # noqa: E501
            )
            self.execute(
                "CREATE TABLE IF NOT EXISTS shape_perceptual_hash_map ( phash_id INTEGER, hash_id INTEGER, PRIMARY KEY ( phash_id, hash_id ) )"  # noqa: E501
            )
            self.execute(
                "CREATE TABLE IF NOT EXISTS shape_vptree ( phash_id INTEGER PRIMARY KEY, parent_id INTEGER, radius INTEGER, inner_id INTEGER, inner_population INTEGER, outer_id INTEGER, outer_population INTEGER )"  # noqa: E501
            )
            # fmt: off
            self.execute(
                "CREATE TABLE IF NOT EXISTS shape_maintenance_branch_regen ( phash_id INTEGER PRIMARY KEY )"
            )
            # fmt: on
            self.execute(
                "CREATE TABLE IF NOT EXISTS shape_search_cache ( hash_id INTEGER PRIMARY KEY, searched_distance INTEGER )"  # noqa: E501
            )

            self.execute(
                "CREATE TABLE IF NOT EXISTS phashed_file_queue ( file_hash BLOB_BYTES NOT NULL UNIQUE, phash BLOB_BYTES NOT NULL, PRIMARY KEY ( file_hash, phash ) )"  # noqa: E501
            )

            # Insert the files from the SqliteDict videos table into the hash queue.
            old_videos_data = []
            print(
                "Migrating perceptually hashed videos from the old table.\n"
                "This may take a bit, depending your db length."
            )

            from pickle import loads

            for key, value in self.execute("SELECT key, value FROM videos"):
                # I don't see why value could be None, but if it happens for whatever reason
                # we just want to continue.
                if value is None:
                    continue
                row = loads(bytes(value))  # this is decode function in SqliteDict
                if "perceptual_hash" in row:
                    video_hash = key
                    old_videos_data.append((video_hash, row["perceptual_hash"]))
                    # The farthest search index will not be moved.

            for video_hash, perceptual_hash in old_videos_data:
                # Add to phashed file queue
                self.execute(
                    "REPLACE INTO phashed_file_queue ( file_hash, phash ) VALUES ( :file_hash, :phash )",
                    {"file_hash": video_hash, "phash": str(perceptual_hash)},
                )

            self.execute("UPDATE version SET version = :version", {"version": "0.7.0"})
            # Note: We need to keep re-running get_version so that we can progressively upgrade.
            version = self.get_version()

        if SemanticVersion(version) < SemanticVersion("0.10.0"):
            print_upgrade(version, "0.10.0")

            print(
                "Migrating perceptually hashed videos from the old format.\n"
                "This may take a bit, depending your db length."
            )

            import json

            def convert_old_vpdq_to_new(old_vpdq_phash_json: str) -> bytes:
                """Convert <0.9.0 full vpdq hash (json) to 0.10.0 hash"""

                # The old vpdq converted the hashes in reverse order. This has no effect on the similarity
                # or anything, but it complicates code in the vpdq implementation, so I changed the order
                # to the native byte order of the PDQ hash.
                def reverse_byte_order(s: str):
                    bytes_obj = bytes.fromhex(s)
                    reversed_bytes = bytes_obj[::-1]
                    return reversed_bytes.hex()

                j = json.loads(old_vpdq_phash_json)
                new_vpdq_hash_str = ""
                for vpdq_feature in j:
                    phash, feature_quality, frame_num = vpdq_feature.split(",")
                    # Filtering hash quality is done during hashing starting in 0.10.0 instead of during similarity
                    # comparison because there's no reason to store hashes which will never be used.
                    if int(feature_quality) >= 31:
                        new_vpdq_hash_str += reverse_byte_order(phash)

                # In the uncommon scenario that all hashes have all been filtered out due to bad quality, then videos
                # will compare as not similar to any other video (including itself, naturally). This matches the same
                # behavior as the old algorithm where low quality hashes were filtered before similarity was calculated.

                return bytes.fromhex(new_vpdq_hash_str)

            # Update phashes in phashed_file_queue
            for phash_id, phash in self.execute("SELECT phash_id, phash FROM shape_perceptual_hashes").fetchall():
                self.execute(
                    "REPLACE INTO shape_perceptual_hashes ( phash_id, phash ) VALUES ( :phash_id, :phash )",
                    {"phash_id": phash_id, "phash": convert_old_vpdq_to_new(phash)},
                )

            # Update phashes in shape_perceptual_hashes
            for file_hash, phash in self.execute("SELECT file_hash, phash FROM phashed_file_queue").fetchall():
                self.execute(
                    "REPLACE INTO phashed_file_queue ( file_hash, phash ) VALUES ( :file_hash, :phash )",
                    {"file_hash": file_hash, "phash": convert_old_vpdq_to_new(phash)},
                )

            self.execute("UPDATE version SET version = :version", {"version": "0.10.0"})
            # Note: We need to keep re-running get_version so that we can progressively upgrade.
            version = self.get_version()

        # No db changes in this case, just print a nice message that your DB is upgraded.
        if SemanticVersion(version) < SemanticVersion(__version__):
            print_upgrade(version, __version__)

        self.set_version(__version__)
        return True


class SemanticVersion:
    """Simple semantic version class. Supports MAJOR.MINOR.PATCH, e.g. 1.2.3"""

    def __init__(self, version: str):
        self.version = version
        try:
            self.parts = list(map(int, version.split(".")))
            if len(self.parts) != 3:
                raise DedupeDbException("len != 3")
        except Exception as exc:
            raise DedupeDbException(f"Bad semantic version: {self.version}.\nFull exception: {exc}")

    def __eq__(self, other):
        return self.parts == other.parts

    def __lt__(self, other):
        return self.parts < other.parts

    def __le__(self, other):
        return self.parts <= other.parts

    def __gt__(self, other):
        return self.parts > other.parts

    def __ge__(self, other):
        return self.parts >= other.parts

    def __repr__(self):
        return f"SemanticVersion('{self.version}')"
