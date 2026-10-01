"""Candidate argument pairs and entity-marker formatting for relation classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bioscrolls import config

# Head/tail order is canonical by type so the classifier sees a consistent layout:
# gene -> chemical -> disease (e.g. gene-disease puts the gene in [E1]).
_TYPE_ORDER = {config.GENE: 0, config.CHEMICAL: 1, config.DISEASE: 2}


@dataclass(frozen=True)
class KeyedMention:
    """A mention span (relative to its sentence) with a normalized entity key."""

    start: int
    end: int
    label: str
    key: str


def candidate_pairs(mentions: Sequence[KeyedMention]) -> list[tuple[KeyedMention, KeyedMention]]:
    """One candidate per in-scope entity pair, using each entity's first mention."""
    first: dict[str, KeyedMention] = {}
    for mention in sorted(mentions, key=lambda m: m.start):
        first.setdefault(mention.key, mention)
    ordered = sorted(first.values(), key=lambda m: (_TYPE_ORDER[m.label], m.start))
    pairs = []
    for i, head in enumerate(ordered):
        for tail in ordered[i + 1 :]:
            if head.key == tail.key:
                continue
            if frozenset({head.label, tail.label}) not in config.RELATION_PAIR_TYPES:
                continue
            pairs.append((head, tail))
    return pairs


def mark_entities(text: str, head: KeyedMention, tail: KeyedMention) -> str:
    """Wrap head/tail spans with ``[E1] .. [/E1]`` and ``[E2] .. [/E2]`` markers."""
    if head.start < tail.end and tail.start < head.end:
        raise ValueError("head and tail spans overlap")
    inserts = sorted(
        [
            (head.start, f"{config.HEAD_START} "),
            (head.end, f" {config.HEAD_END}"),
            (tail.start, f"{config.TAIL_START} "),
            (tail.end, f" {config.TAIL_END}"),
        ],
        key=lambda item: item[0],
    )
    pieces, cursor = [], 0
    for position, marker in inserts:
        pieces.append(text[cursor:position])
        pieces.append(marker)
        cursor = position
    pieces.append(text[cursor:])
    return "".join(pieces)
