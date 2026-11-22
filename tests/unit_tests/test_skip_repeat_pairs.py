from __future__ import annotations

import contextlib
import io
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.db import vptree

from .test_pipeline import THRESHOLD, Recorder, _build, _open, _tables


@contextlib.contextmanager
def _env(**values):
    old = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run(path: Path, pipeline: bool, mark, skip: bool = False, env_skip: str | None = None):
    """Run the real find_potential_duplicates() the way dedup.py does; Hydrus is replaced by `mark`.

    Returns (result, printed_text)."""
    out = io.StringIO()
    with _env(HVD_PIPELINE="1" if pipeline else "0", HVD_SKIP_REPEAT_PAIRS=env_skip):
        db = _open(path)
        d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
        d.db = db
        d.similarity_threshold = THRESHOLD
        d.update_progress_callback = None
        d.should_skip_step_fn = None
        d.mark_videos_as_duplicates = mark
        if skip:
            d.skip_repeat_pairs = True
        try:
            with contextlib.redirect_stdout(out):
                db.begin_transaction()
                with db.conn:
                    try:
                        result = d.find_potential_duplicates()
                    except KeyboardInterrupt:
                        result = "interrupted"
        except Exception as exc:
            result = exc
        finally:
            db.close()
    return result, out.getvalue()


def _reference_calls(path: Path) -> list:
    """The calls the original (pre-pipeline, pre-skip) loop makes, computed straight from the search tree."""
    db = _open(path)
    tree = vptree.VpTreeManager(db)
    threshold = vptree.fix_vpdq_similarity(THRESHOLD)
    files = db.execute(
        "SELECT hash_id FROM shape_search_cache WHERE searched_distance is NULL or searched_distance < :threshold",
        {"threshold": threshold},
    ).fetchall()
    calls = []
    for (hash_id,) in files:
        a = db.get_file_hash(hash_id)
        for similar_hash_id, _distance in tree.search_file(hash_id, max_hamming_distance=threshold):
            if hash_id != similar_hash_id:
                calls.append((a, db.get_file_hash(similar_hash_id)))
    db.close()
    return calls


def _expected_with_skip(calls: list) -> tuple[list, list]:
    """Apply the skip rule to a call list: returns (calls made, index in `calls` of each call made)."""
    sent, made, idx = set(), [], []
    for i, (a, b) in enumerate(calls):
        key = frozenset((a, b))
        if key in sent:
            sent.discard(key)
            continue
        sent.add(key)
        made.append((a, b))
        idx.append(i)
    return made, idx


def _flat(text: str) -> str:
    """Whitespace-insensitive text, so console line wrapping doesn't matter."""
    return " ".join(text.split())


def _pairs(calls) -> set:
    return {frozenset(c) for c in calls if c[0] != "FAILED"}


class TestSkipRepeatPairs(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name) / "base.sqlite"
        _build(self.base)
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def _copy(self, src: Path | None = None) -> Path:
        self.n += 1
        dst = Path(self.tmp.name) / f"c{self.n}.sqlite"
        shutil.copyfile(src or self.base, dst)
        return dst

    def _off(self, pipeline: bool, src: Path | None = None):
        path = self._copy(src)
        rec = Recorder(delay=0)
        result, printed = _run(path, pipeline, rec)
        return path, rec, result, printed

    # 1. Flag off: identical to the current code.
    def test_off_by_default_and_identical(self):
        self.assertFalse(dedup.HydrusVideoDeduplicator(None, client=None).skip_repeat_pairs)
        with _env(HVD_SKIP_REPEAT_PAIRS=None):
            self.assertFalse(dedup.skip_repeat_pairs_enabled_by_env())
        with _env(HVD_SKIP_REPEAT_PAIRS="0"):
            self.assertFalse(dedup.skip_repeat_pairs_enabled_by_env())
        with _env(HVD_SKIP_REPEAT_PAIRS="1"):
            self.assertTrue(dedup.skip_repeat_pairs_enabled_by_env())
        reference = _reference_calls(self._copy())
        self.assertGreater(len(reference), 50)
        self.assertGreater(len(reference) - len(_pairs(reference)), 10)  # the test data really has repeats
        tables = None
        for pipeline in (False, True):
            path, rec, result, printed = self._off(pipeline)
            self.assertEqual(rec.calls, reference)  # same calls, same order
            self.assertEqual(result, len(reference) // 2)
            self.assertNotIn("Skipping repeat pair sends", printed)
            self.assertNotIn("repeat pair sends", printed)
            self.assertNotIn("WARNING:", printed)  # warning is at startup, not here
            self.assertIn(_flat(dedup.SPEED_TIPS), _flat(printed))  # speed tips for everyone
            self.assertEqual(_flat(printed).count("Speed tips:"), 1)  # printed once
            tables = tables or _tables(path)
            self.assertEqual(_tables(path), tables)

    # 2 + 5. Flag on (CLI attribute or env var), with and without the pipeline.
    def test_on_same_pairs_no_repeats_same_order(self):
        for pipeline in (False, True):
            for how in ("flag", "env"):
                with self.subTest(pipeline=pipeline, how=how):
                    off_path, off, off_result, _ = self._off(pipeline)
                    path = self._copy()
                    rec = Recorder(delay=0)
                    if how == "flag":
                        result, printed = _run(path, pipeline, rec, skip=True)
                    else:
                        result, printed = _run(path, pipeline, rec, env_skip="1")
                    expected, _ = _expected_with_skip(off.calls)
                    self.assertEqual(rec.calls, expected)  # original calls, original order, minus repeats
                    self.assertEqual(_pairs(rec.calls), _pairs(off.calls))  # Hydrus gets exactly the same pairs
                    self.assertEqual(len(_pairs(rec.calls)), len(rec.calls))  # no pair sent twice
                    self.assertLess(len(rec.calls), len(off.calls))
                    self.assertEqual(result, off_result)  # printed pair count unchanged
                    self.assertEqual(_tables(path), _tables(off_path))  # 'searched' marks etc. identical
                    # The search-cache warning is printed at startup, not in this phase.
                    self.assertNotIn("WARNING:", printed)
                    # Blank line after the skip-repeat intro (the bar is on stderr).
                    self.assertIn("Skipping repeat pair sends.\n\n", printed)
                    self.assertNotIn(" Using ", printed)
                    if pipeline:
                        self.assertIn("\nUsing pipelined Hydrus calls.\n", printed)
                    else:
                        self.assertNotIn("Using pipelined Hydrus calls.", printed)
                    self.assertIn(_flat(dedup.SPEED_TIPS), _flat(printed))
                    self.assertIn(f"Skipped {len(off.calls) - len(rec.calls)} repeat pair sends", printed)

    # 3. Incremental run: B was searched in an earlier run, new file A finds B -> the pair is still sent.
    def test_incremental_pairs_with_already_searched_files_are_sent(self):
        threshold = vptree.fix_vpdq_similarity(THRESHOLD)
        for pipeline in (False, True):
            with self.subTest(pipeline=pipeline):
                prepared = self._copy()
                db = _open(prepared)
                ids = [h for (h,) in db.execute("SELECT hash_id FROM shape_search_cache ORDER BY rowid").fetchall()]
                # every other file was "searched in an earlier run"
                for h in ids[::2]:
                    db.execute("UPDATE shape_search_cache SET searched_distance = ? WHERE hash_id = ?", (threshold, h))
                db.commit()
                old = {db.get_file_hash(h) for h in ids[::2]}
                db.close()

                off_path, off, off_result, _ = self._off(pipeline, prepared)
                path = self._copy(prepared)
                rec = Recorder(delay=0)
                result, _ = _run(path, pipeline, rec, skip=True)
                cross = [c for c in off.calls if c[1] in old]
                self.assertGreater(len(cross), 5)  # new file finds an old, already searched file
                for c in cross:
                    self.assertIn(c, rec.calls)
                self.assertEqual(_pairs(rec.calls), _pairs(off.calls))
                self.assertEqual(rec.calls, _expected_with_skip(off.calls)[0])
                self.assertEqual(result, off_result)
                self.assertEqual(_tables(path), _tables(off_path))

    # 4. A Hydrus error mid-file: same error handling as the original, the failed pair is not recorded as sent.
    def test_hydrus_error_mid_run(self):
        _, full, _, _ = self._off(False)
        made, idx = _expected_with_skip(full.calls)
        for pipeline in (False, True):
            for k in (1, len(made) // 3, len(made) // 2, len(made) - 1):
                with self.subTest(pipeline=pipeline, k=k):
                    a_path, b_path = self._copy(), self._copy()
                    orig = Recorder(fail_at=idx[k], delay=0)  # original code failing on the same request
                    skip = Recorder(fail_at=k, delay=0)
                    ra, _ = _run(a_path, pipeline, orig)
                    rb, printed = _run(b_path, pipeline, skip, skip=True)
                    self.assertIsInstance(ra, ConnectionError)
                    self.assertIsInstance(rb, ConnectionError)  # same error surfaces
                    self.assertEqual(skip.calls[-1], ("FAILED",) + made[k])
                    self.assertEqual(_tables(b_path), _tables(a_path))  # same files marked, rest rolled back
                    self.assertIn("repeat pair sends", printed)  # end summary still printed

                    # Next run (flag on) finishes the job: final DB identical, every pair reached Hydrus.
                    again = Recorder(delay=0)
                    _run(b_path, pipeline, again, skip=True)
                    self.assertEqual(_tables(b_path), _tables(self._off(pipeline)[0]))
                    self.assertEqual(_pairs(skip.calls) | _pairs(again.calls), _pairs(full.calls))

    def test_failed_send_is_not_recorded(self):
        calls = []

        def send(a, b):
            calls.append((a, b))
            if len(calls) == 1:
                raise ConnectionError("simulated Hydrus failure")

        skipper = dedup.RepeatPairSkipper(send)
        a, b = "a" * 64, "b" * 64
        with self.assertRaises(ConnectionError):
            skipper(a, b)
        skipper(b, a)  # not skipped: the first send never succeeded
        skipper(a, b)  # skipped: (b, a) was sent successfully
        skipper(b, a)  # sent again: a third occurrence is never skipped twice
        self.assertEqual(calls, [(a, b), (b, a), (b, a)])
        self.assertEqual(skipper.skipped, 1)
        self.assertEqual(len(skipper._sent), 1)

    def test_search_phase_console_spacing(self):
        # Skip-repeat off: blank line after tips and after Using lines, no double gaps, no leading spaces.
        path, rec, result, printed = self._off(True)
        self.assertIn("subscriptions.\n\nUsing ", printed)
        # The bar writes to stderr, so stdout should end with a blank line after the Using block.
        self.assertTrue(printed.endswith("Using pipelined Hydrus calls.\n\n"), printed[-80:])
        self.assertNotIn(" Using ", printed)
        self.assertNotIn("\n\n\n", printed)
        self.assertNotIn("Skipping repeat pair sends", printed)
        self.assertNotIn("WARNING:", printed)

        # Skip-repeat on: Using lines, blank, skip-info line, blank, then the bar (stderr).
        path = self._copy()
        _result, printed = _run(path, True, Recorder(delay=0), skip=True)
        self.assertIn("Using pipelined Hydrus calls.\nSkipping repeat pair sends.\n\n", printed)
        self.assertNotIn("WARNING:", printed)
        self.assertNotIn(" Using ", printed)
        self.assertNotIn("\n\n\n", printed)

    def test_startup_text(self):
        self.assertEqual(dedup.SKIP_REPEAT_PAIRS_INFO, "Skipping repeat pair sends.")
        self.assertEqual(
            dedup.SEARCH_CACHE_WARNING,
            (
                'WARNING: If you clear or resync potential duplicates in Hydrus ("delete all potential duplicate pairs and re-search" '
                "or \"resync potential duplicate pairs to storage\"), you will have to clear this tool's search cache "
                "(--clear-search-cache) and re-search every video from zero. A large library can take several hours or days."
            ),
        )
        self.assertEqual(
            dedup.SPEED_TIPS,
            (
                "Speed tips: The final phase of the tool is limited by Hydrus's database queues.\n"
                "The fewer jobs Hydrus has queued, the faster this phase runs:\n"
                " - Close any Hydrus duplicates pages (they recount pairs as they arrive).\n"
                " - Pause duplicate auto-resolution rules.\n"
                " - Pause Hydrus idle/background maintenance.\n"
                " - Pause active downloads and subscriptions."
            ),
        )
        for line in dedup.SPEED_TIPS.split("\n"):
            self.assertLessEqual(len(line), 80)

    def test_no_files_to_search_prints_no_tips(self):
        path = self._copy()
        db = _open(path)
        db.execute("UPDATE shape_search_cache SET searched_distance = ?", (vptree.fix_vpdq_similarity(THRESHOLD),))
        db.commit()
        db.close()
        rec = Recorder(delay=0)
        result, printed = _run(path, True, rec)
        self.assertEqual((result, rec.calls), (0, []))
        self.assertNotIn("Speed tips", printed)
        self.assertNotIn("WARNING:", printed)
        self.assertNotIn("Skipping repeat pair sends", printed)

    def test_memory_entry_dropped_after_repeat(self):
        skipper = dedup.RepeatPairSkipper(lambda a, b: None)
        h = [f"{i:064x}" for i in range(4)]
        skipper(h[0], h[1])
        skipper(h[2], h[3])
        self.assertEqual(len(skipper._sent), 2)
        skipper(h[1], h[0])
        skipper(h[3], h[2])
        self.assertEqual(len(skipper._sent), 0)
        self.assertEqual(skipper.skipped, 2)
        # non-hex / odd hashes still work (fallback key)
        skipper("x", "y")
        skipper("y", "x")
        self.assertEqual(skipper.skipped, 3)


if __name__ == "__main__":
    unittest.main()
