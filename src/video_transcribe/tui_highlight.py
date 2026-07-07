"""Regex-driven Rich highlighter for the TUI log pane.

Tuned to what this pipeline actually prints: the CLI's ``==>`` / indented step
lines, ``[MM:SS]`` transcript timestamps, ``wrote <path>`` confirmations,
percentages, ``error:`` / ``warning:`` diagnostics, and the doctor's status
markers. Rules are applied in order; later rules override earlier ones where
ranges overlap, so specific tokens win over broad ones.

Colours are bright pastels chosen to stay readable on a dark terminal and on
Textual's selection tint (full-saturation primaries collide with it).
"""

from __future__ import annotations

import re
from typing import Final

from rich.text import Text

__all__ = ("highlight_line", "is_warning_line")

_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Absolute paths ending in a media/text extension we produce or consume.
    (re.compile(
        r"(?:[A-Za-z]:[\\/]|/)[\w\\/.\- ]+"
        r"\.(?:mp4|mkv|m4a|mp3|wav|mov|webm|txt|srt|vtt|json)\b",
        re.IGNORECASE,
    ), "cyan"),
    # http(s) URLs (the diarization/token help lines are full of them).
    (re.compile(r"\bhttps?://[\w\-./?=&%#:+~]+", re.IGNORECASE), "cyan underline"),
    # Transcript / progress timestamps: [MM:SS] or [H:MM:SS], and bare MM:SS/HH:MM:SS.
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"), "italic #7AE0E5"),
    # Percentages in the progress bar.
    (re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%"), "bold #7AE582"),
    # The CLI's step markers: leading "==>" and the 4-space indented sub-steps.
    (re.compile(r"^\s*==>"), "bold #E09BFF"),
    # Severity tokens (error must win over warning on an "error: ... warning" line).
    (re.compile(r"(?<!_)\b(?:warning|warn)\b\s*:?", re.IGNORECASE), "bold #FFD862"),
    (re.compile(r"(?<!_)\b(?:error|fatal)\b\s*:?", re.IGNORECASE), "bold #FF8A8A"),
    # Doctor status markers.
    (re.compile(r"\[ok\s*\]", re.IGNORECASE), "bold #7AE582"),
    (re.compile(r"\[FAIL\]"), "bold #FF8A8A"),
    (re.compile(r"\bwrote\b", re.IGNORECASE), "bold #7AE582"),
    # Our own [tui]/[key] prefixes and the "--- exit_code=... ---" summary.
    (re.compile(r"^\[[\w\-]+\]"), "bold #E09BFF"),
    (re.compile(r"---\s*exit_code=.*?---"), "bold"),
)

_WARNING_RE: Final = re.compile(r"\bwarning\b", re.IGNORECASE)
_ZERO_WARNING_RE: Final = re.compile(r"\b0\s+warning", re.IGNORECASE)


def is_warning_line(line: str) -> bool:
    """Whether a committed log line should count toward a task's warning tally
    (drives the amber 'warned' badge). '0 warnings' is not a warning."""
    return bool(_WARNING_RE.search(line)) and not _ZERO_WARNING_RE.search(line)


def highlight_line(line: str) -> Text:
    text = Text(line)
    for pattern, style in _RULES:
        for match in pattern.finditer(line):
            text.stylize(style, match.start(), match.end())
    return text
