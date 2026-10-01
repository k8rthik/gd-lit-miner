"""Offset-preserving sentence splitter tuned for biomedical abstracts.

Rule based on purpose: it is deterministic, dependency free, and returns
character offsets into the original string so entity spans stay aligned.
"""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "vs",
        "cf",
        "fig",
        "figs",
        "al",
        "et al",
        "approx",
        "ca",
        "no",
        "nos",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "resp",
        "ref",
        "refs",
        "eq",
        "vol",
        "sp",
        "spp",
        "inc",
        "ltd",
        "co",
        "st",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "p",
    }
)
# Candidate boundary: terminal punctuation (optionally closing quote/bracket) + whitespace.
_BOUNDARY_RE = re.compile(r"[.!?][\"')\]]?\s+")
_PREV_TOKEN_RE = re.compile(r"(\S+)$")


def _is_abbreviation(text: str, punct_index: int) -> bool:
    if text[punct_index] != ".":
        return False
    match = _PREV_TOKEN_RE.search(text[:punct_index])
    if not match:
        return False
    token = match.group(1).lstrip("([").lower()
    return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _starts_sentence(text: str, index: int) -> bool:
    if index >= len(text):
        return False
    char = text[index]
    return char.isupper() or char in "([\"'"


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of sentences in ``text`` (whitespace trimmed)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        next_start = match.end()
        if not _starts_sentence(text, next_start) or _is_abbreviation(text, match.start()):
            continue
        spans.append((start, match.end()))
        start = next_start
    spans.append((start, len(text)))
    return [trimmed for s, e in spans if (trimmed := _trim(text, s, e)) is not None]


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None
