from __future__ import annotations

import logging
import os
import sys

from rich import print

# Printed once at startup when below-normal priority is applied (display only).
BELOW_NORMAL_PRIORITY_INFO = (
    "Running at below-normal priority with reduced CPU use. Add --full-speed to use all CPU cores."
)

# Windows SetPriorityClass value. Below-normal yields to other apps but still runs nearly
# full speed when the PC is idle. Idle priority is too aggressive and can starve the tool.
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
# Windows memory priority (SetProcessInformation ProcessMemoryPriority): when RAM runs short, Windows reuses this
# process's cached pages before other apps'. Measured: no slowdown.
# Not PROCESS_MODE_BACKGROUND_BEGIN: it also caps the working set at about 32 MiB, which made hashing a 1080p video
# about 14 times slower (millions of page faults).
PROCESS_MEMORY_PRIORITY = 0  # PROCESS_INFORMATION_CLASS ProcessMemoryPriority
MEMORY_PRIORITY_LOW = 2

_ON_VALUES = ("1", "true", "yes", "on")

_log = logging.getLogger(__name__)
_applied = False
_info_pending = False


def full_speed_requested(full_speed: bool = False) -> bool:
    """True for full-speed mode: --full-speed (or its old name --normal-priority), or HVD_FULL_SPEED or
    HVD_NORMAL_PRIORITY on. Full speed: all CPU cores, normal priority, more videos downloaded ahead.
    Otherwise (the default): reduced CPU use, below-normal priority, fewer videos downloaded ahead."""
    if full_speed:
        return True
    return any(
        os.environ.get(name, "0").strip().lower() in _ON_VALUES for name in ("HVD_FULL_SPEED", "HVD_NORMAL_PRIORITY")
    )


# Old name, kept for --normal-priority / HVD_NORMAL_PRIORITY (now the same as --full-speed).
normal_priority_requested = full_speed_requested


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


def _windows_set_low_memory_priority() -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    set_process_information = kernel32.SetProcessInformation
    set_process_information.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    set_process_information.restype = wintypes.BOOL
    value = wintypes.ULONG(MEMORY_PRIORITY_LOW)
    if not set_process_information(
        get_current_process(), PROCESS_MEMORY_PRIORITY, ctypes.byref(value), ctypes.sizeof(value)
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _set_below_normal_priority() -> None:
    if sys.platform == "win32":
        _windows_set_priority_class(BELOW_NORMAL_PRIORITY_CLASS)
        try:
            _windows_set_low_memory_priority()
        except Exception:  # e.g. Windows 7: below-normal CPU priority still applies
            _log.debug("Could not set low memory priority", exc_info=True)
    else:
        os.nice(10)


def apply_process_priority(*, full_speed: bool = False, normal_priority: bool = False) -> bool:
    """Lower this process to below-normal priority (and low memory priority on Windows) unless in full-speed mode.

    Returns True if priority was lowered. Failures are logged at debug and ignored.
    Child threads and subprocesses inherit the process priority.
    """
    global _applied, _info_pending
    if _applied:
        return False
    if full_speed_requested(full_speed or normal_priority):
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
    print(f"[bright_blue]{BELOW_NORMAL_PRIORITY_INFO}")
    return True
