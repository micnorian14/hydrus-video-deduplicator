from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

    # The logging level for logging e.g. CRITICAL or WARNING
    Severity: TypeAlias = int

from rich import print


def severity_to_color(severity: Severity) -> str:
    if severity > logging.WARNING:
        return "[red]"
    elif severity == logging.WARNING:
        return "[yellow]"
    elif severity <= logging.INFO:
        return ""  # No color
    # No color
    return ""


def print_and_log(logger: logging.Logger, msg: str, severity: Severity = logging.INFO):
    """
    Print to the user and log. Changes print color based on the severity.

    The default logger uses the logging module.
    """
    print(f"{severity_to_color(severity)}{msg}")
    logger.log(severity, msg)


def display_path(path, *, windows: bool | None = None, environ=None, home: str | None = None) -> str:
    """
    A path for PRINTING only, with the user-specific start masked so screenshots don't show the user name.

    Windows: the %LOCALAPPDATA% folder (else the %USERPROFILE% folder) is shown as %LOCALAPPDATA% / %USERPROFILE%
    (case-insensitive). When %LOCALAPPDATA% is the usual %USERPROFILE%\\AppData\\Local it is shown that way instead,
    so users can find the AppData\\Local folder. Other systems: the home folder is shown as ~. Any other path is
    returned unchanged.
    Never use the result to open or create anything.
    """
    text = str(path)
    if windows is None:
        windows = os.name == "nt"
    if environ is None:
        environ = os.environ
    if windows:
        lowered = text.lower()
        for var in ("LOCALAPPDATA", "USERPROFILE"):
            base = (environ.get(var) or "").rstrip("\\/")
            if len(base) < 3:  # unset, or just a drive like "C:"
                continue
            low_base = base.lower()
            if lowered == low_base or lowered.startswith(low_base + "\\") or lowered.startswith(low_base + "/"):
                profile = (environ.get("USERPROFILE") or "").rstrip("\\/")
                if var == "LOCALAPPDATA" and len(profile) >= 3 and low_base == (profile + "\\AppData\\Local").lower():
                    return "%USERPROFILE%" + text[len(profile):]  # keeps the path's own AppData\Local casing
                return f"%{var}%" + text[len(base):]
        return text
    if home is None:
        home = os.path.expanduser("~")
    home = home.rstrip("/")
    if home and (text == home or text.startswith(home + "/")):
        return "~" + text[len(home):]
    return text
