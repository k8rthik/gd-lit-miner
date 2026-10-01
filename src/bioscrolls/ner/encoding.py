"""Pure conversions between character spans and BIO token tags.

Kept free of tokenizer objects: callers pass the tokenizer's ``offset_mapping``,
which makes these functions easy to test and reuse at train and inference time.
Special tokens have offsets ``(0, 0)`` and get ``IGNORE_INDEX``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from bioscrolls.config import NER_LABELS
from bioscrolls.models import Span

IGNORE_INDEX = -100
LABEL_TO_ID = MappingProxyType({label: i for i, label in enumerate(NER_LABELS)})
_OUTSIDE = LABEL_TO_ID["O"]

Offsets = Sequence[tuple[int, int]]


def _is_special(offset: tuple[int, int]) -> bool:
    return offset[0] == offset[1]


def spans_to_token_tags(offsets: Offsets, spans: Sequence[Span]) -> list[int]:
    """Tag each token B-/I-<type> if it overlaps a span, else O."""
    tags: list[int] = []
    previous_span: Span | None = None
    for offset in offsets:
        if _is_special(offset):
            tags.append(IGNORE_INDEX)
            previous_span = None
            continue
        start, end = offset
        span = next((s for s in spans if start < s.end and end > s.start), None)
        if span is None:
            tags.append(_OUTSIDE)
        elif span is previous_span:
            tags.append(LABEL_TO_ID[f"I-{span.label}"])
        else:
            tags.append(LABEL_TO_ID[f"B-{span.label}"])
        previous_span = span
    return tags


def decode_tags(offsets: Offsets, tag_ids: Sequence[int], text: str) -> list[Span]:
    """Merge B/I token runs into character spans (lenient: orphan I- starts a span)."""
    if len(offsets) != len(tag_ids):
        raise ValueError(f"{len(offsets)} offsets but {len(tag_ids)} tags")
    spans: list[Span] = []
    current: tuple[int, int, str] | None = None
    for offset, tag_id in zip(offsets, tag_ids, strict=True):
        if _is_special(offset) or tag_id == IGNORE_INDEX:
            continue
        tag = NER_LABELS[tag_id]
        prefix, _, label = tag.partition("-")
        continues = current is not None and prefix == "I" and label == current[2]
        if continues:
            current = (current[0], offset[1], label)
            continue
        if current is not None:
            spans.append(Span(current[0], current[1], current[2], text[current[0] : current[1]]))
        current = (offset[0], offset[1], label) if prefix in ("B", "I") else None
    if current is not None:
        spans.append(Span(current[0], current[1], current[2], text[current[0] : current[1]]))
    return spans


def remove_overlaps(spans: Sequence[Span]) -> list[Span]:
    """Keep a non-overlapping subset, preferring longer spans then earlier ones."""
    ranked = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
    kept: list[Span] = []
    for span in ranked:
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)
