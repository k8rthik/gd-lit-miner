"""Lexical normalisation keys used to match mentions against vocabularies.

Keys are case-folded, Greek letters are spelled out, possessives and
punctuation are removed and tokens are sorted, so that "alpha-synuclein",
"α-synuclein" and "synuclein alpha" collapse to the same key.
"""

from __future__ import annotations

import re
import unicodedata

_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "τ": "tau",
    "ω": "omega",
}
_POSSESSIVE_RE = re.compile(r"['’`]s\b")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_MIN_PLURAL_LENGTH = 4


def normalize_key(text: str) -> str:
    lowered = "".join(_GREEK.get(ch, ch) for ch in text.lower())
    lowered = unicodedata.normalize("NFKD", lowered)
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    lowered = _POSSESSIVE_RE.sub("", lowered)
    tokens = _NON_ALNUM_RE.sub(" ", lowered).split()
    return " ".join(sorted(tokens))


def _singular(token: str) -> str:
    if len(token) < _MIN_PLURAL_LENGTH:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def lookup_keys(text: str) -> list[str]:
    """Keys to try, most specific first: the exact key, then a singularised key."""
    key = normalize_key(text)
    singular = " ".join(sorted(_singular(t) for t in key.split()))
    return [key] if singular == key else [key, singular]
