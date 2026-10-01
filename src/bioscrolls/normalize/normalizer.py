"""Map NER mentions to vocabulary identifiers (dictionary-based normalisation).

Strategy, in order: (1) expand a known abbreviation (in-document definition,
else the most frequent corpus-wide definition) and look up the long form;
(2) look up the mention itself; (3) drop a parenthetical short form inside the
mention. Lookups use order/case/punctuation-insensitive keys plus a
singularised variant. Unmatched mentions keep a deterministic text-based id so
they still aggregate across documents, but are flagged ``matched=False``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bioscrolls.normalize.abbreviations import find_abbreviations
from bioscrolls.normalize.keys import lookup_keys, normalize_key
from bioscrolls.normalize.lexicon import LexiconEntry, build_index

_PAREN_RE = re.compile(r"\s*\([^()]*\)")


@dataclass(frozen=True)
class Resolution:
    entity_id: str
    name: str
    matched: bool


def candidate_keys(surface: str, abbreviations: Mapping[str, str]) -> list[str]:
    keys: list[str] = []
    long_form = abbreviations.get(surface.strip())
    if long_form:
        keys.extend(lookup_keys(long_form))
    keys.extend(lookup_keys(surface))
    stripped = _PAREN_RE.sub("", surface)
    if stripped != surface:
        keys.extend(lookup_keys(stripped))
    return list(dict.fromkeys(k for k in keys if k))


def corpus_abbreviations(texts: Iterable[str]) -> dict[str, str]:
    """Most frequent long form for each short form across a corpus."""
    votes: dict[str, Counter] = {}
    surfaces: dict[tuple[str, str], str] = {}
    for text in texts:
        for short, long_form in find_abbreviations(text).items():
            key = normalize_key(long_form)
            votes.setdefault(short, Counter())[key] += 1
            surfaces.setdefault((short, key), long_form)
    return {short: surfaces[(short, counts.most_common(1)[0][0])] for short, counts in votes.items()}


class Normalizer:
    def __init__(self, indices: Mapping[str, Mapping[str, tuple[str, str]]]) -> None:
        self._indices = {label: dict(index) for label, index in indices.items()}

    @classmethod
    def build(
        cls,
        sources: Mapping[str, Iterable[LexiconEntry]],
        mentions: Iterable[tuple[str, str, Mapping[str, str]]],
    ) -> Normalizer:
        """Build per-type indices restricted to keys these mentions could need."""
        wanted: dict[str, set[str]] = {label: set() for label in sources}
        for surface, label, abbreviations in mentions:
            wanted.setdefault(label, set()).update(candidate_keys(surface, abbreviations))
        return cls({label: build_index(entries, wanted[label]) for label, entries in sources.items()})

    def resolve(self, surface: str, label: str, abbreviations: Mapping[str, str]) -> Resolution:
        index = self._indices.get(label, {})
        keys = candidate_keys(surface, abbreviations)
        for key in keys:
            if key in index:
                entity_id, name = index[key]
                return Resolution(entity_id, name, True)
        fallback = keys[-1] if keys else surface.lower()
        return Resolution(f"{label.lower()}-text:{fallback}", surface, False)
