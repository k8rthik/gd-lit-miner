"""BioRED corpus (Luo et al., 2022) in PubTator format.

BioRED provides 600 PubMed abstracts (400 train / 100 dev / 100 test) with
normalized entity annotations and *document-level* relation annotations
between entity identifiers. We keep Gene, Disease and Chemical entities and
the cross-type relations among them.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from bioscrolls import config

log = logging.getLogger(__name__)

_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
_MISSING_IDS = frozenset({"", "-"})


class CorpusFormatError(ValueError):
    """Raised when a corpus file does not match the expected format."""


@dataclass(frozen=True)
class GoldEntity:
    start: int
    end: int
    text: str
    label: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class GoldRelation:
    id1: str
    id2: str
    label: str


@dataclass(frozen=True)
class AnnotatedDoc:
    pmid: str
    title: str
    abstract: str
    entities: tuple[GoldEntity, ...]
    relations: tuple[GoldRelation, ...]

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}"

    @property
    def title_length(self) -> int:
        return len(self.title)

    def relation_between(self, a: str, b: str) -> str | None:
        for rel in self.relations:
            if {rel.id1, rel.id2} == {a, b}:
                return rel.label
        return None

    def entity_types(self) -> dict[str, str]:
        return {i: e.label for e in self.entities for i in e.ids}


def _parse_entity(fields: list[str], text: str) -> GoldEntity | None:
    _, start, end, surface, raw_type, raw_ids = fields
    label = config.BIORED_ENTITY_MAP.get(raw_type)
    if label is None:
        return None
    s, e = int(start), int(end)
    if text[s:e] != surface:
        raise CorpusFormatError(f"offset mismatch for {surface!r}: found {text[s:e]!r}")
    ids = tuple(i for i in raw_ids.split(",") if i not in _MISSING_IDS)
    return GoldEntity(s, e, surface, label, ids)


def _parse_block(block: str) -> AnnotatedDoc:
    lines = block.strip("\n").split("\n")
    if len(lines) < 2 or "|t|" not in lines[0] or "|a|" not in lines[1]:
        raise CorpusFormatError("document block must start with |t| and |a| lines")
    pmid, title = lines[0].split("|t|", 1)
    _, abstract = lines[1].split("|a|", 1)
    text = f"{title} {abstract}"
    entities: list[GoldEntity] = []
    raw_relations: list[tuple[str, str, str]] = []
    for line in lines[2:]:
        fields = line.split("\t")
        if len(fields) == 6:
            entity = _parse_entity(fields, text)
            if entity is not None:
                entities.append(entity)
        elif len(fields) == 5:
            raw_relations.append((fields[1], fields[2], fields[3]))
        else:
            raise CorpusFormatError(f"malformed line in {pmid}: {line[:80]!r}")
    types = {i: e.label for e in entities for i in e.ids}
    relations = tuple(
        GoldRelation(a, b, config.BIORED_RELATION_MAP[rtype])
        for rtype, a, b in raw_relations
        if rtype in config.BIORED_RELATION_MAP
        and a in types
        and b in types
        and frozenset({types[a], types[b]}) in config.RELATION_PAIR_TYPES
    )
    return AnnotatedDoc(pmid, title, abstract, tuple(entities), relations)


def parse_pubtator(raw: str) -> list[AnnotatedDoc]:
    blocks = [b for b in _BLOCK_SPLIT_RE.split(raw) if b.strip()]
    return [_parse_block(b) for b in blocks]


def load_split(split: str, corpus_dir: Path = config.CORPORA_DIR) -> list[AnnotatedDoc]:
    if split not in config.BIORED_SPLITS:
        raise ValueError(f"unknown BioRED split {split!r}; expected one of {config.BIORED_SPLITS}")
    path = Path(corpus_dir) / "BioRED" / f"{split}.PubTator"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bioscrolls download-corpus` first")
    return parse_pubtator(path.read_text(encoding="utf-8"))


def download_biored(corpus_dir: Path = config.CORPORA_DIR, session: requests.Session | None = None) -> Path:
    """Download and unpack the BioRED release (~2 MB). Idempotent."""
    target = Path(corpus_dir) / "BioRED"
    if all((target / f"{s}.PubTator").exists() for s in config.BIORED_SPLITS):
        return target
    http = session or requests.Session()
    response = http.get(config.BIORED_URL, timeout=config.HTTP_TIMEOUT_S)
    if response.status_code != 200:
        raise RuntimeError(f"BioRED download failed: HTTP {response.status_code}")
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        members = [m for m in archive.namelist() if m.endswith(".PubTator")]
        archive.extractall(Path(corpus_dir), members=members)
    log.info("BioRED extracted to %s", target)
    return target
