"""Schwartz & Hearst (2003) abbreviation detection: ``long form (SF)`` patterns.

Used to resolve short forms such as "AD" to "Alzheimer's disease" within the
same abstract before vocabulary lookup.
"""

from __future__ import annotations

import re

_PAREN_RE = re.compile(r"\(([^()]{1,12})\)")
_MAX_SHORT_WORDS = 2


def _is_short_form(candidate: str) -> bool:
    if not 2 <= len(candidate) <= 10 or len(candidate.split()) > _MAX_SHORT_WORDS:
        return False
    return candidate[0].isalnum() and any(c.isalpha() for c in candidate)


def _best_long_form(short: str, preceding: str) -> str | None:
    """Scan right-to-left matching short-form characters inside the preceding words."""
    words = preceding.split()
    max_words = min(len(short) + 5, len(short) * 2)
    window = " ".join(words[-max_words:])
    s_index, l_index = len(short) - 1, len(window) - 1
    while s_index >= 0:
        char = short[s_index].lower()
        if not char.isalnum():
            s_index -= 1
            continue
        while l_index >= 0 and (
            window[l_index].lower() != char or (s_index == 0 and l_index > 0 and window[l_index - 1].isalnum())
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    start = window.rfind(" ", 0, l_index + 1) + 1
    long_form = window[start:].strip()
    if len(long_form) <= len(short) or short.lower() in long_form.lower().split():
        return None
    return long_form


def find_abbreviations(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for match in _PAREN_RE.finditer(text):
        short = match.group(1).strip()
        if not _is_short_form(short) or short in found:
            continue
        long_form = _best_long_form(short, text[: match.start()])
        if long_form:
            found[short] = long_form
    return found
