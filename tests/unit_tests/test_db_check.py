from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from hydrusvideodeduplicator import db_tools
from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes
from .test_orphans import _break
from .test_relink import _new_db, _tables


def _one(db, sql, args=()):
    return db.execute(sql, args).fetchone()[0]


def _inner_node(db) -> int:
    """A non-root node that has a child (so cutting its link detaches a branch)."""
    return _one(
        db,
        "SELECT phash_id FROM shape_vptree WHERE parent_id IS NOT NULL AND inner_id IS NOT NULL "
        "ORDER BY phash_id LIMIT 1;",
    )


def _leaf(db) -> int:
    return _one(
        db,
        "SELECT phash_id FROM shape_vptree WHERE parent_id IS NOT NULL AND inner_id IS NULL AND outer_id IS NULL "
        "ORDER BY phash_id LIMIT 1;",
    )


def _unlink(db, phash_id: int):
    db.execute("UPDATE shape_vptree SET inner_id = NULL WHERE inner_id = ?;", (phash_id,))
    db.execute("UPDATE shape_vptree SET outer_id = NULL WHERE outer_id = ?;", (phash_id,))


def _detach_branch(db):
    _unlink(db, _inner_node(db))


def _delete_node(db):
    db.execute("DELETE FROM shape_vptree WHERE phash_id = ?;", (_inner_node(db),))


def _loop(db):
    leaf, ancestor = _leaf(db), _one(db, "SELECT inner_id FROM shape_vptree WHERE parent_id IS NULL;")
    db.execute("UPDATE shape_vptree SET inner_id = ? WHERE phash_id = ?;", (ancestor, leaf))


def _second_root(db):
    db.execute("UPDATE shape_vptree SET parent_id = NULL WHERE phash_id = ?;", (_inner_node(db),))


def _leaf_missing_from_tree(db):
    leaf = _leaf(db)
    _unlink(db, leaf)
    db.execute("DELETE FROM shape_vptree WHERE phash_id = ?;", (leaf,))


DAMAGE = {
    "detached branch": (_detach_branch, "unreachable"),
    "missing node": (_delete_node, "broken_links"),
    "loop": (_loop, "shared"),
    "second root": (_second_root, "extra_roots"),
    "hash missing from tree": (_leaf_missing_from_tree, "not_in_tree"),
}


class TestSearchTreeCheck(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _db(self, tmpdir, seed=11):
        return _new_db(tmpdir, "check.sqlite", _make_hashes(seed=seed))

    def assertHealthy(self, db):
        result = db.check_search_tree()
        self.assertEqual(result["problems"], 0, result)
        return result

    def test_healthy_trees(self):
        with TemporaryDirectory() as tmpdir:
            db = self._db(tmpdir)  # a normal tree, built one file at a time like "Building search tree"
            result = self.assertHealthy(db)
            self.assertEqual(result["nodes"], _one(db, "SELECT COUNT(*) FROM shape_perceptual_hashes;"))
            db.execute("INSERT INTO shape_maintenance_branch_regen ( phash_id ) VALUES ( ? );", (_inner_node(db),))
            self.assertHealthy(db)  # pending branch maintenance is normal
            _break(db)  # broken entries are a different check
            self.assertHealthy(db)
            db.commit()
            db.begin_transaction()
            with db.conn:
                db.relink_search_tree()  # the chain tree --relink-search-tree builds
            self.assertHealthy(db)
            keep = [f for (f,) in db.execute("SELECT file_hash FROM files ORDER BY hash_id;")][10:]
            db.begin_transaction()
            with db.conn:
                db.prune_files_not_in(keep)  # --prune-deleted-files splices nodes out
            self.assertHealthy(db)
            db.close()

    def test_empty_database(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "empty.sqlite", [])
            self.assertEqual(db.check_search_tree()["nodes"], 0)
            self.assertHealthy(db)
            db.close()

    def test_tree_cleared_or_still_being_built_is_not_damage(self):
        with TemporaryDirectory() as tmpdir:
            db = self._db(tmpdir)
            db.begin_transaction()
            with db.conn:
                db.clear_search_tree()  # every file goes back to the queue; the tree is empty
            self.assertEqual(db.check_search_tree()["nodes"], 0)
            self.assertHealthy(db)
            queue = db.execute("SELECT file_hash, phash FROM phashed_file_queue ORDER BY file_hash;").fetchall()
            for file_hash, phash in queue[:20]:  # "Building search tree" interrupted part way
                db.add_file(file_hash)
                db.add_perceptual_hash(phash)
                db.associate_file_with_perceptual_hash(file_hash, phash)
                db.execute("DELETE FROM phashed_file_queue WHERE file_hash = ?;", (file_hash,))
            db.commit()
            self.assertGreater(db.check_search_tree()["nodes"], 0)
            self.assertHealthy(db)
            db.close()

    def test_each_kind_of_damage_is_found_and_relink_repairs_it(self):
        for name, (damage, key) in DAMAGE.items():
            with self.subTest(name), TemporaryDirectory() as tmpdir:
                db = self._db(tmpdir)
                damage(db)
                db.commit()
                result = db.check_search_tree()
                self.assertGreater(result[key], 0, result)
                self.assertGreater(result["problems"], 0)
                db.begin_transaction()
                with db.conn:
                    db.relink_search_tree()
                self.assertHealthy(db)
                db.close()

    def test_queued_file_is_not_flagged(self):
        with TemporaryDirectory() as tmpdir:
            db = self._db(tmpdir)
            leaf = _leaf(db)
            _leaf_missing_from_tree(db)
            self.assertEqual(db.check_search_tree()["not_in_tree"], 1)
            for (file_hash,) in db.execute(
                "SELECT file_hash FROM files JOIN shape_perceptual_hash_map USING ( hash_id ) WHERE phash_id = ?;",
                (leaf,),
            ).fetchall():
                db.add_to_phashed_files_queue(file_hash, db.get_phash(leaf))
            self.assertHealthy(db)
            db.close()

    def test_tree_node_without_hash(self):
        with TemporaryDirectory() as tmpdir:
            db = self._db(tmpdir)
            db.execute("DELETE FROM shape_perceptual_hashes WHERE phash_id = ?;", (_leaf(db),))
            self.assertEqual(db.check_search_tree()["no_hash"], 1)
            db.close()


def _tree(**counts) -> dict:
    tree = dict.fromkeys(
        ("extra_roots", "no_root", "broken_links", "shared", "unreachable", "wrong_parent", "no_hash", "not_in_tree"), 0
    )
    nodes = counts.pop("nodes", 393057)
    tree.update(counts)
    tree["problems"] = sum(tree.values())
    tree["nodes"] = nodes
    return tree


FIX = (
    "Run once with --relink-search-tree to repair it. Your database is backed up automatically. "
    "The next duplicate search will start over."
)


class TestTreeMessages(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(
            db_tools.tree_check_message(_tree()), "Search tree: 393057 perceptual hashes, no search tree issues found."
        )
        self.assertEqual(
            db_tools.tree_check_message(_tree(nodes=1)), "Search tree: 1 perceptual hash, no search tree issues found."
        )

    def test_every_issue_plural(self):
        tree = _tree(unreachable=193057, broken_links=2, shared=5, extra_roots=2, wrong_parent=4, no_hash=3)
        tree["not_in_tree"] = 12
        tree["problems"] += 12
        self.assertEqual(
            db_tools.tree_check_message(tree).splitlines(),
            [
                "WARNING: Search tree: 393057 perceptual hashes, issues found:",
                "  - 193057 hashes unreachable by the search (cut-off branches or loops)",
                "  - 2 links point to nodes that don't exist",
                "  - 5 nodes are linked from more than one parent",
                "  - 3 top nodes found (should be 1)",
                "  - 4 parent links don't match",
                "  - 3 tree nodes have no hash",
                "  - 12 hashes are missing from the tree",
                FIX,
            ],
        )

    def test_every_issue_singular(self):
        tree = _tree(unreachable=1, broken_links=1, shared=1, extra_roots=1, wrong_parent=1, no_hash=1, not_in_tree=1)
        self.assertEqual(
            db_tools.tree_check_message(tree).splitlines()[1:-1],
            [
                "  - 1 hash unreachable by the search (cut-off branches or loops)",
                "  - 1 link points to a node that doesn't exist",
                "  - 1 node is linked from more than one parent",
                "  - 2 top nodes found (should be 1)",
                "  - 1 parent link doesn't match",
                "  - 1 tree node has no hash",
                "  - 1 hash is missing from the tree",
            ],
        )

    def test_only_nonzero_lines_and_no_root(self):
        lines = db_tools.tree_check_message(_tree(no_root=1, unreachable=10)).splitlines()
        self.assertEqual(
            lines,
            [
                "WARNING: Search tree: 393057 perceptual hashes, issues found:",
                "  - 10 hashes unreachable by the search (cut-off branches or loops)",
                "  - no top node found (should be 1)",
                FIX,
            ],
        )
        self.assertEqual(
            db_tools.tree_check_message(_tree(not_in_tree=2)).splitlines()[1:-1],
            ["  - 2 hashes are missing from the tree"],
        )

    def test_gui_last_line(self):
        lines = db_tools.tree_check_message(_tree(broken_links=2), gui=True).splitlines()
        self.assertEqual(lines[1], "  - 2 links point to nodes that don't exist")
        self.assertEqual(
            lines[2],
            "Some videos may never be compared. To repair it, run the command line version once with "
            "--relink-search-tree. Your database is backed up automatically. "
            "The next duplicate search will start over.",
        )


class TestStartupLines(unittest.TestCase):
    def setUp(self):
        vptree.TemporaryIntegerTableNameCache()

    def _lines(self, db, **kwargs):
        with (
            mock.patch.object(db_tools, "print_success") as green,
            mock.patch.object(db_tools, "print_and_log") as warning,
        ):
            db_tools.check_database(db, **kwargs)
        return [("green", c.args[0]) for c in green.call_args_list] + [
            ("warning", c.args[1]) for c in warning.call_args_list
        ]

    def test_lines(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "lines.sqlite", _make_hashes(seed=12))
            files, nodes = db.count_files(), _one(db, "SELECT COUNT(*) FROM shape_vptree;")
            healthy = [
                ("green", f"Checked {files} files: no broken entries."),
                ("green", f"Search tree: {nodes} perceptual hashes, no search tree issues found."),
            ]
            self.assertEqual(self._lines(db), healthy)
            self.assertEqual(self._lines(db, check_orphans=False), healthy[1:])
            self.assertEqual(self._lines(db, check_tree=False), healthy[:1])

            _break(db)  # 3 broken entries, 1 more file with a valid (empty) hash
            files, nodes = db.count_files(), _one(db, "SELECT COUNT(*) FROM shape_vptree;")
            lines = self._lines(db)
            clean_tree = f"Search tree: {nodes} perceptual hashes, no search tree issues found."
            self.assertEqual(lines[0], ("green", clean_tree))
            self.assertEqual(lines[1], ("warning", db_tools.orphans_check_message(files, 3)))
            self.assertTrue(lines[1][1].startswith(f"WARNING: Checked {files} files: 3 videos are missing their"))

            _detach_branch(db)
            db.commit()
            before = _tables(db)
            tree = db.check_search_tree()
            lines = self._lines(db, gui=True)
            self.assertEqual([kind for kind, _ in lines], ["warning", "warning"])
            self.assertEqual(lines[0][1], db_tools.orphans_check_message(files, 3, gui=True))
            self.assertEqual(lines[1][1], db_tools.tree_check_message(tree, gui=True))
            self.assertIn("hashes unreachable by the search", lines[1][1])
            self.assertEqual(_tables(db), before)  # read only
            self.assertFalse(db.conn.in_transaction)
            db.close()

    def test_tree_damage_only(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "lines.sqlite", _make_hashes(seed=13))
            _detach_branch(db)
            db.commit()
            lines = self._lines(db)
            self.assertEqual(lines[0], ("green", f"Checked {db.count_files()} files: no broken entries."))
            self.assertEqual(lines[1][0], "warning")
            self.assertIn("Run once with --relink-search-tree to repair it", lines[1][1])
            db.close()

    def test_a_failing_check_never_stops_a_run(self):
        with TemporaryDirectory() as tmpdir:
            db = _new_db(tmpdir, "lines.sqlite", _make_hashes(seed=14))
            with mock.patch.object(DedupeDB.DedupeDb, "check_search_tree", side_effect=RuntimeError("boom")):
                lines = self._lines(db)
            self.assertEqual([kind for kind, _ in lines], ["green"])
            db.close()


if __name__ == "__main__":
    unittest.main()
