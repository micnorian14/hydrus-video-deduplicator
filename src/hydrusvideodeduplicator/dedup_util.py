from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

    # The logging level for logging e.g. CRITICAL or WARNING
    Severity: TypeAlias = int

from rich import get_console
from rich.text import Text
from tqdm import tqdm

# Cargo / rustc / git convention (display only): the label is colored, the message is plain.
# WARNING: uses red3 (#d70000), not bold — bold red looks pinkish in Windows Terminal.
STYLE_TIP_LABEL = "bold cyan"
STYLE_WARNING_LABEL = "red3"
STYLE_ERROR_LABEL = "red"
STYLE_CODE = "magenta"  # flags, env vars, key combos, like Markdown inline code
STYLE_WARN_LINE = "yellow"  # whole-line warnings with no natural prefix label

# Longer labels first so "Speed tips:" is not split by "Tip:".
# "ERROR " matches "ERROR clearing..." and colors only the word ERROR.
_LABELED_PREFIXES = (
    ("Speed tips:", STYLE_TIP_LABEL),
    ("WARNING:", STYLE_WARNING_LABEL),
    ("Tip:", STYLE_TIP_LABEL),
    ("ERROR:", STYLE_ERROR_LABEL),
    ("ERROR ", STYLE_ERROR_LABEL),
)

# CLI flags, env vars, and CTRL+C. Does not match quoted Hydrus menu item names.
CODE_TOKEN_RE = re.compile(
    r"(?:CTRL\+C|Ctrl\+C|"
    r"HVD_[A-Z0-9_]+(?:=[A-Za-z0-9_.-]+)?|"
    r"HYDRUS_[A-Z0-9_]+(?:=[A-Za-z0-9_.-]+)?|"
    r"--[A-Za-z0-9][A-Za-z0-9\-]*(?:=[A-Za-z0-9_.-]+)?)"
)


def style_code_spans(text: str, base_style: str | None = None) -> Text:
    """Plain text (or base_style) with flags, env vars, and key combos in magenta. Brackets are literal."""
    out = Text()
    pos = 0
    for match in CODE_TOKEN_RE.finditer(text):
        if match.start() > pos:
            out.append(text[pos:match.start()], style=base_style)
        out.append(match.group(0), style=STYLE_CODE)
        pos = match.end()
    if pos < len(text):
        out.append(text[pos:], style=base_style)
    return out


def style_user_text(s: str, *, default_style: str | None = None) -> Text:
    """Style a user-facing message: known labels colored, flags magenta, everything else plain or default_style."""
    out = Text()
    for i, line in enumerate(s.split("\n")):
        if i:
            out.append("\n")
        out.append_text(_style_line(line, default_style))
    return out


def _style_line(line: str, default_style: str | None) -> Text:
    body = line.lstrip(" ")
    leading = line[: len(line) - len(body)]
    for label, style in _LABELED_PREFIXES:
        if body.startswith(label):
            colored = label.rstrip()  # "ERROR " -> "ERROR"; "WARNING:" unchanged
            rest = body[len(colored):]
            out = Text(leading)
            out.append(colored, style=style)
            out.append_text(style_code_spans(rest))
            return out
    out = Text(leading, style=default_style)
    out.append_text(style_code_spans(body, base_style=default_style))
    return out


def console_print(*args, highlight: bool = False, **kwargs) -> None:
    """Rich Console.print. highlight defaults to off so magenta flags and literal brackets are not restyled."""
    get_console().print(*args, highlight=highlight, **kwargs)


def print_user(s: str, *, default_style: str | None = None) -> None:
    """Print a styled user message without Rich auto-highlighting, so magenta flags and literal brackets stay intact."""
    console_print(style_user_text(s, default_style=default_style), highlight=False)


def print_over_bar(*args, **kwargs) -> None:
    """Rich print, clearing any active tqdm bar first so the message starts on its own line."""
    kwargs.setdefault("highlight", False)
    instances = getattr(tqdm, "_instances", None)
    inst = instances[0] if instances else None
    if inst is not None:
        with inst.external_write_mode(file=inst.fp):
            console_print(*args, **kwargs)
    else:
        console_print(*args, **kwargs)


def severity_to_color(severity: Severity) -> str:
    if severity > logging.WARNING:
        return "[red]"
    elif severity == logging.WARNING:
        return "[yellow]"
    elif severity <= logging.INFO:
        return ""  # No color
    # No color
    return ""


class TqdmLoggingHandler(logging.StreamHandler):
    """A StreamHandler that writes through tqdm so log lines start on their own line if a progress bar is active."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def print_and_log(logger: logging.Logger, msg: str, severity: Severity = logging.INFO):
    """
    Print to the user and log. Changes print color based on the severity.

    The default logger uses the logging module.
    Unlabeled error/warning lines stay whole-line red/yellow; labels like WARNING: use the cargo convention.
    Flags in the text are magenta. The logged message is unchanged.
    """
    if severity > logging.WARNING:
        default_style = STYLE_ERROR_LABEL
    elif severity == logging.WARNING:
        default_style = STYLE_WARN_LINE
    else:
        default_style = None
    print_user(msg, default_style=default_style)
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
