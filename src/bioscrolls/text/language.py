"""Cheap English/non-English guess for abstracts (stopword ratio).

PubMed's ``Language`` field describes the *article*, not the abstract we receive:
non-English articles usually carry an English abstract. Our NER/RE models are
English-only, so we check the abstract text itself before processing it.
"""

from __future__ import annotations

import re

ENGLISH = "en"
NON_ENGLISH = "non-en"
UNKNOWN = "unknown"

_MIN_TOKENS = 8
_ENGLISH_RATIO_THRESHOLD = 0.12
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_ENGLISH_STOPWORDS = frozenset(
    "the of and in to a is was were with for that by as on are this from be an or "
    "we which these at not have has been than between our its their it after".split()
)


def guess_language(text: str) -> str:
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    if len(tokens) < _MIN_TOKENS:
        return UNKNOWN
    ratio = sum(t in _ENGLISH_STOPWORDS for t in tokens) / len(tokens)
    return ENGLISH if ratio >= _ENGLISH_RATIO_THRESHOLD else NON_ENGLISH
