from __future__ import annotations

import logging
import os
import sys

from rich import print

# Printed once at startup when below-normal priority is applied (display only).
BELOW_NORMAL_PRIORITY_INFO = "Running at below-normal priority. Add --normal-priority to run at full priority."

# Windows SetPriorityClass value. Below-normal yields to other apps but still runs nearly
# full speed when the PC is idle. Idle priority is too aggressive and can starve the tool.
BELOW_NORMAL_PRIORITY_CLASS = 0x4000

_log = logging.getLogger(__name__)
_applied = False
_info_pending = False


def normal_priority_requested(normal_priority: bool = False) -> bool:
    """True if --normal-priority was passed or HVD_NORMAL_PRIORITY is on."""
    if normal_priority:
        return True
    return os.environ.get("HVD_NORMAL_PRIORITY", "0").strip().lower() in ("1", "true", "yes", "on")


def _windows_set_priority_class(priority_class: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    set_priority_class = kernel32.SetPriorityClass
    set_priority_class.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    set_priority_class.restype = wintypes.BOOL
    if not set_priority_class(get_current_process(), priority_class):
        raise ctypes.WinError(ctypes.get_last_error())


def _set_below_normal_priority() -> None:
    if sys.platform == "win32":
        _windows_set_priority_class(BELOW_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(10)


def apply_process_priority(*, normal_priority: bool = False) -> bool:
    """Lower this process to below-normal priority unless asked not to.

    Returns True if priority was lowered. Failures are logged at debug and ignored.
    Child threads and subprocesses inherit the process priority.
    """
    global _applied, _info_pending
    if _applied:
        return False
    if normal_priority_requested(normal_priority):
        _applied = True
        return False
    try:
        _set_below_normal_priority()
    except Exception:
        _log.debug("Could not set below-normal process priority", exc_info=True)
        _applied = True
        return False
    _applied = True
    _info_pending = True
    return True


def print_priority_info() -> bool:
    """Print the below-normal priority line once, if priority was lowered. Returns True if printed."""
    global _info_pending
    if not _info_pending:
        return False
    _info_pending = False
    print(f"[blue]{BELOW_NORMAL_PRIORITY_INFO}")
    return True
