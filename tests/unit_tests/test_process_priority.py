from __future__ import annotations

import io
import logging
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

from hydrusvideodeduplicator import process_priority as pp


class TestProcessPriority(unittest.TestCase):
    def setUp(self):
        pp._applied = False
        pp._info_pending = False
        self._old_env = os.environ.get("HVD_NORMAL_PRIORITY")
        os.environ.pop("HVD_NORMAL_PRIORITY", None)

    def tearDown(self):
        pp._applied = False
        pp._info_pending = False
        if self._old_env is None:
            os.environ.pop("HVD_NORMAL_PRIORITY", None)
        else:
            os.environ["HVD_NORMAL_PRIORITY"] = self._old_env

    def test_default_lowers_priority_and_prints_info(self):
        with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
            with redirect_stdout(io.StringIO()) as out:
                lowered = pp.apply_process_priority()
        set_prio.assert_called_once_with()
        self.assertTrue(lowered)
        # The line is printed later, next to the GPU decode line.
        self.assertEqual(out.getvalue(), "")
        with redirect_stdout(io.StringIO()) as out:
            self.assertTrue(pp.print_priority_info())
        self.assertIn(pp.BELOW_NORMAL_PRIORITY_INFO, out.getvalue())

    def test_flag_skips_lowering(self):
        with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
            with redirect_stdout(io.StringIO()) as out:
                lowered = pp.apply_process_priority(normal_priority=True)
        set_prio.assert_not_called()
        self.assertFalse(lowered)
        self.assertEqual(out.getvalue(), "")

    def test_env_var_skips_lowering(self):
        os.environ["HVD_NORMAL_PRIORITY"] = "1"
        with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
            with redirect_stdout(io.StringIO()) as out:
                lowered = pp.apply_process_priority()
        set_prio.assert_not_called()
        self.assertFalse(lowered)
        self.assertEqual(out.getvalue(), "")

    def test_env_var_aliases(self):
        for value in ("true", "YES", "On"):
            with self.subTest(value=value):
                pp._applied = False
                os.environ["HVD_NORMAL_PRIORITY"] = value
                with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
                    self.assertFalse(pp.apply_process_priority())
                set_prio.assert_not_called()

    def test_env_var_off_still_lowers(self):
        os.environ["HVD_NORMAL_PRIORITY"] = "0"
        with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
            self.assertTrue(pp.apply_process_priority())
        set_prio.assert_called_once_with()

    def test_failure_is_silent_and_does_not_raise(self):
        with mock.patch.object(pp, "_set_below_normal_priority", side_effect=OSError("denied")):
            with redirect_stdout(io.StringIO()) as out:
                lowered = pp.apply_process_priority()
        self.assertFalse(lowered)
        self.assertEqual(out.getvalue(), "")

    def test_failure_is_logged_at_debug(self):
        with mock.patch.object(pp, "_set_below_normal_priority", side_effect=OSError("denied")):
            with self.assertLogs(pp._log, level=logging.DEBUG) as captured:
                pp.apply_process_priority()
        self.assertTrue(any("Could not set below-normal process priority" in line for line in captured.output))

    def test_second_call_does_not_set_or_print_again(self):
        with mock.patch.object(pp, "_set_below_normal_priority") as set_prio:
            with redirect_stdout(io.StringIO()) as out:
                self.assertTrue(pp.apply_process_priority())
                self.assertFalse(pp.apply_process_priority())
                self.assertTrue(pp.print_priority_info())
                self.assertFalse(pp.print_priority_info())
        set_prio.assert_called_once_with()
        self.assertEqual(out.getvalue().count(pp.BELOW_NORMAL_PRIORITY_INFO), 1)

    def test_no_info_line_when_priority_not_lowered(self):
        with mock.patch.object(pp, "_set_below_normal_priority"):
            pp.apply_process_priority(normal_priority=True)
        with redirect_stdout(io.StringIO()) as out:
            self.assertFalse(pp.print_priority_info())
        self.assertEqual(out.getvalue(), "")

    def test_windows_sets_below_normal_priority_class(self):
        with mock.patch.object(pp.sys, "platform", "win32"):
            with mock.patch.object(pp, "_windows_set_priority_class") as set_class:
                with mock.patch.object(pp.os, "nice", create=True) as nice:
                    pp._set_below_normal_priority()
        set_class.assert_called_once_with(0x4000)
        nice.assert_not_called()

    def test_linux_and_macos_use_nice(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with mock.patch.object(pp.sys, "platform", platform):
                    with mock.patch.object(pp, "_windows_set_priority_class") as set_class:
                        with mock.patch.object(pp.os, "nice", create=True) as nice:
                            pp._set_below_normal_priority()
                set_class.assert_not_called()
                nice.assert_called_once_with(10)

    def test_windows_set_priority_class_calls_kernel32(self):
        ctypes_mod = mock.MagicMock()
        wintypes = mock.MagicMock()
        ctypes_mod.wintypes = wintypes
        kernel32 = mock.MagicMock()
        kernel32.GetCurrentProcess.return_value = 42
        kernel32.SetPriorityClass.return_value = 1
        ctypes_mod.WinDLL.return_value = kernel32
        with mock.patch.dict(sys.modules, {"ctypes": ctypes_mod, "ctypes.wintypes": wintypes}):
            pp._windows_set_priority_class(pp.BELOW_NORMAL_PRIORITY_CLASS)
        ctypes_mod.WinDLL.assert_called_once_with("kernel32", use_last_error=True)
        kernel32.SetPriorityClass.assert_called_once_with(42, 0x4000)

    def test_windows_set_priority_class_failure_raises(self):
        ctypes_mod = mock.MagicMock()
        wintypes = mock.MagicMock()
        ctypes_mod.wintypes = wintypes
        kernel32 = mock.MagicMock()
        kernel32.GetCurrentProcess.return_value = 42
        kernel32.SetPriorityClass.return_value = 0
        ctypes_mod.WinDLL.return_value = kernel32
        ctypes_mod.get_last_error.return_value = 5
        ctypes_mod.WinError.side_effect = OSError("access denied")
        with mock.patch.dict(sys.modules, {"ctypes": ctypes_mod, "ctypes.wintypes": wintypes}):
            with self.assertRaises(OSError):
                pp._windows_set_priority_class(pp.BELOW_NORMAL_PRIORITY_CLASS)
