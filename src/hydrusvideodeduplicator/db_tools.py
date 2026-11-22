"""--backup, --prune-deleted-files and --prune-orphans (all off by default), shared by the CLI and the GUI."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import TYPE_CHECKING

from rich import print

from .db import DedupeDB
from .dedup_util import STYLE_WARN_LINE, display_path, print_and_log, print_success, print_user

if TYPE_CHECKING:
    from pathlib import Path

    from .client import HVDClient

log = logging.getLogger("db_tools")


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


def backup_message(path) -> str:
    return f"Backed up the database to: {display_path(path)}"


def prune_message(result: DedupeDB.PruneResult) -> str:
    if result.videos == 0:
        return "No deleted videos to prune."
    return (
        f"Pruned {result.videos} deleted {_plural(result.videos, 'video')} from the database and "
        f"{result.tree_nodes} {_plural(result.tree_nodes, 'entry', 'entries')} from the search tree."
    )


def deleted_backup_message(path) -> str:
    return f"Deleted the oldest backup: {display_path(path)}"


def delete_backup_failed_message(path, exc: OSError) -> str:
    return f"Could not delete the oldest backup: {display_path(path)}: {exc.strerror or exc}"


def run_backup(db_path: Path | None = None) -> Path | None:
    """
    --backup: back up the database (if it exists) and print where the copy went, and which old backups were
    deleted to keep the 3 newest. Raises if the backup fails; a failed delete of an old backup is only a warning.
    """
    removed: list = []
    backup_path = DedupeDB.backup_db(db_path, removed=removed)
    if backup_path is not None:
        log.info(f"Backed up the database to {backup_path}")
        print_success(backup_message(backup_path))
    for old, exc in removed:  # backup_db already logged these
        if exc is None:
            print_user(deleted_backup_message(old), highlight=False)
        else:
            print_user(delete_backup_failed_message(old, exc), default_style=STYLE_WARN_LINE, highlight=False)
    return backup_path


def run_prune(db: DedupeDB.DedupeDb, client: HVDClient) -> DedupeDB.PruneResult | None:
    """
    --prune-deleted-files: remove videos Hydrus no longer has from the dedupe database (take a backup first).

    Asks Hydrus once for every video it has (read only). If that fails or returns no videos, nothing is pruned.
    All-or-nothing: on an error or CTRL+C the database is left unchanged. Returns None if nothing was pruned.
    """
    print("[blue]Checking Hydrus for deleted videos...")
    try:
        existing = client.get_all_video_hashes_in_hydrus()
    except Exception as exc:
        reason = re.sub(r"[0-9a-fA-F]{64}", "<hidden>", str(exc))  # never show an API key or file hash
        print_and_log(
            log,
            f"Skipping --prune-deleted-files: could not get the video list from Hydrus ({reason}).",
            logging.WARNING,
            highlight=False,
        )
        return None
    if not existing:
        print_and_log(
            log, "Skipping --prune-deleted-files: Hydrus returned no videos, so nothing was pruned.", logging.WARNING
        )
        return None

    try:
        db.begin_transaction()
        with db.conn:  # all-or-nothing: if anything fails or is interrupted, the DB is left unchanged
            result = db.prune_files_not_in(existing)
    except KeyboardInterrupt:
        print("[yellow]Pruning was interrupted! Nothing was pruned.")
        return None
    log.info(f"Pruned {result}")
    print_success(prune_message(result))
    return result



def orphans_check_message(files: int, broken: int, gui: bool = False) -> str:
    checked = f"Checked {files} {_plural(files, 'file')}"
    if broken == 0:
        return f"{checked}: no broken entries."
    missing = "1 video is missing its" if broken == 1 else f"{broken} videos are missing their"
    found = f"WARNING: {checked}: {missing} perceptual hash."
    it, they = ("it", "It is") if broken == 1 else ("them", "They are")
    if gui:
        return (
            f"{found} {they} skipped, so nothing breaks. To remove {it}, run the command line version once "
            "with --prune-orphans. Your database is backed up automatically."
        )
    return f"{found} Run once with --prune-orphans to remove {it}. Your database is backed up automatically."


def _tree_issue_lines(tree: dict[str, int]) -> list[str]:
    """One line per kind of search tree damage that was found (zero counts are left out)."""
    lines = []

    def add(count: int, one: str, many: str):
        if count > 0:
            lines.append(one if count == 1 else f"{count} {many}")

    add(tree["unreachable"], "1 hash unreachable by the search", "hashes unreachable by the search")
    if lines:
        lines[-1] += " (cut-off branches or loops)"
    add(tree["broken_links"], "1 link points to a node that doesn't exist", "links point to nodes that don't exist")
    add(tree["shared"], "1 node is linked from more than one parent", "nodes are linked from more than one parent")
    if tree["no_root"]:
        lines.append("no top node found (should be 1)")
    elif tree["extra_roots"]:
        lines.append(f"{tree['extra_roots'] + 1} top nodes found (should be 1)")
    add(tree["wrong_parent"], "1 parent link doesn't match", "parent links don't match")
    add(tree["no_hash"], "1 tree node has no hash", "tree nodes have no hash")
    add(tree["not_in_tree"], "1 hash is missing from the tree", "hashes are missing from the tree")
    return lines


def tree_check_message(tree: dict[str, int], gui: bool = False) -> str:
    """tree is the result of DedupeDb.check_search_tree()."""
    header = f"Search tree: {tree['nodes']} perceptual {_plural(tree['nodes'], 'hash', 'hashes')}"
    if tree["problems"] == 0:
        return f"{header}, no search tree issues found."
    if gui:
        fix = (
            "Some videos may never be compared. To repair it, run the command line version once with "
            "--relink-search-tree. Your database is backed up automatically. The next duplicate search will start over."
        )
    else:
        fix = (
            "Run once with --relink-search-tree to repair it. Your database is backed up automatically. "
            "The next duplicate search will start over."
        )
    issues = [f"  - {line}" for line in _tree_issue_lines(tree)]
    return "\n".join([f"WARNING: {header}, issues found:", *issues, fix])


def prune_orphans_message(count: int) -> str:
    if count == 0:
        return "No broken entries to remove."
    return f"Removed {count} broken {_plural(count, 'entry', 'entries')} (they will be re-hashed on the next run)."


def check_database(
    db: DedupeDB.DedupeDb, check_orphans: bool = True, check_tree: bool = True, gui: bool = False
) -> dict[str, int]:
    """
    Startup checks (read only): one line for broken entries (--prune-orphans fixes them) and one for the search
    tree (--relink-search-tree fixes it). Green when clean, a red WARNING when not. Skip a check when its fix flag
    is given. A failing check is only logged; it never stops a run. Returns the counts that were checked.
    """
    counts = {}
    if check_orphans:
        try:
            counts["files"] = db.count_files()
            counts["broken_entries"] = db.count_orphaned_files()
        except sqlite3.Error as exc:
            log.warning(f"Could not check the database for broken entries: {exc}")
        else:
            _print_check(orphans_check_message(counts["files"], counts["broken_entries"], gui))
    if check_tree:
        try:
            tree = db.check_search_tree()
        except Exception as exc:
            log.warning(f"Could not check the search tree: {exc!r}")
        else:
            log.info(f"Search tree check: {tree}")
            counts["tree_nodes"], counts["tree_problems"] = tree["nodes"], tree["problems"]
            _print_check(tree_check_message(tree, gui))
    return counts


def _print_check(message: str) -> None:
    if message.startswith("WARNING:"):
        print_and_log(log, message, logging.WARNING, highlight=False)
    else:
        log.info(message)
        print_success(message)


def run_prune_orphans(db: DedupeDB.DedupeDb) -> int:
    """
    --prune-orphans: remove broken entries (files whose perceptual hash is missing) so the next run re-hashes them.
    All-or-nothing: on an error or CTRL+C the database is left unchanged. Take a backup first.
    """
    try:
        db.begin_transaction()
        with db.conn:
            removed = db.remove_orphaned_files()
    except KeyboardInterrupt:
        print("[yellow]Removing broken entries was interrupted! Nothing was removed.")
        return 0
    log.info(f"Removed {removed} broken entries")
    print_success(prune_orphans_message(removed))
    return removed
