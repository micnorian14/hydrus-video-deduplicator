"""--backup and --prune-deleted-files (both off by default), shared by the CLI and the GUI."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rich import print

from .db import DedupeDB
from .dedup_util import display_path, print_and_log, print_success

if TYPE_CHECKING:
    from pathlib import Path

    from .client import HVDClient

log = logging.getLogger("db_tools")


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or f"{singular}s")


def backup_message(path) -> str:
    return f"Backed up the database to '{display_path(path)}'."


def prune_message(result: DedupeDB.PruneResult) -> str:
    if result.videos == 0:
        return "No deleted videos to prune."
    return (
        f"Pruned {result.videos} deleted {_plural(result.videos, 'video')} from the database and "
        f"{result.tree_nodes} {_plural(result.tree_nodes, 'entry', 'entries')} from the search tree."
    )


def run_backup(db_path: Path | None = None) -> Path | None:
    """--backup: back up the database (if it exists) and print where the copy went. Raises if the backup fails."""
    backup_path = DedupeDB.backup_db(db_path)
    if backup_path is not None:
        log.info(f"Backed up the database to {backup_path}")
        print_success(backup_message(backup_path))
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
