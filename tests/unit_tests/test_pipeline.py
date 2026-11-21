from __future__ import annotations

import _thread
import os
import shutil
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydrusvideodeduplicator import dedup
from hydrusvideodeduplicator.db import DedupeDB, vptree

from .test_fast_search import _make_hashes

THRESHOLD = 50.0  # similarity %, the app default


def _build(path: Path):
    db = DedupeDB.DedupeDb(path.parent, path.name)
    db.init_connection()
    db.create_tables()
    for i, phash in enumerate([h for h in _make_hashes(seed=31) if h] + [h for h in _make_hashes(seed=32) if h]):
        file_hash = f"{i:064x}"
        db.add_file(file_hash)
        db.add_perceptual_hash(phash)
        db.associate_file_with_perceptual_hash(file_hash, phash)
    db.commit()
    db.close()


def _open(path: Path) -> DedupeDB.DedupeDb:
    vptree.TemporaryIntegerTableNameCache()
    db = DedupeDB.DedupeDb(path.parent, path.name)
    db.init_connection()
    return db


def _tables(path: Path) -> dict:
    db = _open(path)
    names = [n for (n,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;").fetchall()]
    out = {n: db.execute(f"SELECT * FROM {n} ORDER BY rowid;").fetchall() for n in names}
    db.close()
    return out


def _run(path: Path, pipeline: bool, mark) -> object:
    """Run the real find_potential_duplicates() the way dedup.py does; Hydrus is replaced by `mark`."""
    old = os.environ.get("HVD_PIPELINE")
    os.environ["HVD_PIPELINE"] = "1" if pipeline else "0"
    db = _open(path)
    d = dedup.HydrusVideoDeduplicator.__new__(dedup.HydrusVideoDeduplicator)
    d.db = db
    d.similarity_threshold = THRESHOLD
    d.update_progress_callback = None
    d.should_skip_step_fn = None
    d.mark_videos_as_duplicates = mark
    try:
        db.begin_transaction()
        with db.conn:
            try:
                return d.find_potential_duplicates()
            except KeyboardInterrupt:
                return "interrupted"
    except Exception as exc:
        return exc
    finally:
        db.close()
        if old is None:
            del os.environ["HVD_PIPELINE"]
        else:
            os.environ["HVD_PIPELINE"] = old


class Recorder:
    def __init__(self, fail_at=None, interrupt_at=None, delay=0.002):
        self.calls = []
        self.threads = set()
        self.fail_at = fail_at
        self.interrupt_at = interrupt_at
        self.delay = delay

    def __call__(self, a, b):
        self.threads.add(threading.current_thread().name)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            self.calls.append(("FAILED", a, b))
            raise ConnectionError("simulated Hydrus failure")
        time.sleep(self.delay)
        self.calls.append((a, b))
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            _thread.interrupt_main()  # simulated Ctrl+C, arrives in the main thread wherever it is
            time.sleep(0.05)


class TestPipelinedHydrusCalls(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name) / "base.sqlite"
        _build(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def _copy(self, name: str) -> Path:
        dst = Path(self.tmp.name) / name
        shutil.copyfile(self.base, dst)
        return dst

    def _original(self):
        path = self._copy("orig.sqlite")
        rec = Recorder()
        result = _run(path, False, rec)
        self.assertGreater(len(rec.calls), 50)
        return path, rec, result

    def test_identical_calls_order_and_tables(self):
        orig_path, orig, orig_result = self._original()
        self.assertEqual(orig.threads, {"MainThread"})
        path = self._copy("pipe.sqlite")
        rec = Recorder()
        result = _run(path, True, rec)
        self.assertEqual(rec.threads, {"hvd-hydrus-sender"})
        self.assertEqual(rec.calls, orig.calls)  # same requests, same order
        self.assertEqual(result, orig_result)  # same pair count
        self.assertEqual(_tables(path), _tables(orig_path))

    def test_hydrus_error_mid_run(self):
        _, orig, _ = self._original()
        for fail_at in (1, len(orig.calls) // 2, len(orig.calls) - 1):
            a_path, b_path = self._copy(f"a{fail_at}.sqlite"), self._copy(f"b{fail_at}.sqlite")
            a, b = Recorder(fail_at=fail_at), Recorder(fail_at=fail_at)
            ra, rb = _run(a_path, False, a), _run(b_path, True, b)
            self.assertIsInstance(ra, ConnectionError)
            self.assertIsInstance(rb, ConnectionError)  # same error surfaces
            self.assertEqual(b.calls, a.calls)
            self.assertEqual(_tables(b_path), _tables(a_path), f"fail_at {fail_at}")

    def test_ctrl_c_then_resume(self):
        orig_path, orig, _ = self._original()
        full = orig.calls
        for interrupt_at in (3, len(full) // 2):
            path = self._copy(f"cc{interrupt_at}.sqlite")
            db = _open(path)
            files = [
                db.get_file_hash(h)
                for (h,) in db.execute(
                    "SELECT hash_id FROM shape_search_cache WHERE searched_distance is NULL or searched_distance < ?",
                    (vptree.fix_vpdq_similarity(THRESHOLD),),
                ).fetchall()
            ]
            db.close()
            first = Recorder(interrupt_at=interrupt_at)
            self.assertEqual(_run(path, True, first), "interrupted")
            sent = first.calls
            self.assertEqual(sent, full[: len(sent)])  # a prefix of exactly the same calls

            db = _open(path)
            marked = {
                db.get_file_hash(h)
                for (h,) in db.execute(
                    "SELECT hash_id FROM shape_search_cache WHERE searched_distance IS NOT NULL"
                ).fetchall()
            }
            db.close()
            m = len(marked)
            self.assertEqual(marked, set(files[:m]))  # only the first m files, in order
            done_calls = [c for c in full if c[0] in marked]
            self.assertEqual(sent[: len(done_calls)], done_calls)  # every marked file got all its calls
            self.assertLess(m, len(files))

            second = Recorder()
            _run(path, True, second)
            # resume re-sends the first unmarked file from its start, exactly like an interrupted file today
            self.assertEqual(second.calls, full[len(done_calls) :])
            self.assertEqual(_tables(path), _tables(orig_path))

    def test_can_be_disabled(self):
        old = os.environ.get("HVD_PIPELINE")
        os.environ["HVD_PIPELINE"] = "0"
        try:
            self.assertFalse(dedup.pipeline_enabled())
        finally:
            if old is None:
                del os.environ["HVD_PIPELINE"]
            else:
                os.environ["HVD_PIPELINE"] = old
        self.assertTrue(dedup.pipeline_enabled() or os.environ.get("HVD_PIPELINE") == "0")


if __name__ == "__main__":
    unittest.main()
