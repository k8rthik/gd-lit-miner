"""Readers for the normalisation vocabularies (HGNC genes, CTD diseases/chemicals).

Vocabularies are streamed and filtered to the keys actually needed, so the
40 MB CTD chemical vocabulary never has to sit fully in memory.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import requests

from bioscrolls import config
from bioscrolls.normalize.keys import lookup_keys

log = logging.getLogger(__name__)

PREFERRED, SYNONYM, ALIAS = 0, 1, 2
_SINGULAR_PENALTY = 3  # singularised vocabulary keys rank below any exact key
HGNC_FILE = "hgnc_complete_set.txt"
CTD_DISEASES_FILE = "CTD_diseases.tsv.gz"
CTD_CHEMICALS_FILE = "CTD_chemicals.tsv.gz"


class LexiconFormatError(ValueError):
    """Raised when a vocabulary file lacks the expected columns."""


@dataclass(frozen=True)
class LexiconEntry:
    surface: str
    entity_id: str
    name: str
    priority: int


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _split_list(value: str) -> list[str]:
    return [v.strip() for v in value.strip().strip('"').split("|") if v.strip()]


def iter_hgnc(path: Path) -> Iterator[LexiconEntry]:
    """Yield approved symbols/names (preferred), previous symbols and aliases.
    IDs are NCBI Gene IDs where HGNC provides one (matches BioRED), else HGNC IDs."""
    with _open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        required = {"hgnc_id", "symbol", "name", "alias_symbol", "alias_name", "prev_symbol", "entrez_id"}
        if not required <= col.keys():
            raise LexiconFormatError(f"HGNC file missing columns: {sorted(required - col.keys())}")
        for line in handle:
            row = line.rstrip("\n").split("\t")
            row += [""] * (len(header) - len(row))
            symbol = row[col["symbol"]]
            entrez = row[col["entrez_id"]].strip()
            entity_id = f"NCBIGene:{entrez}" if entrez else row[col["hgnc_id"]]
            yield LexiconEntry(symbol, entity_id, symbol, PREFERRED)
            yield LexiconEntry(row[col["name"]], entity_id, symbol, PREFERRED)
            for field, priority in (
                ("prev_symbol", SYNONYM),
                ("alias_symbol", ALIAS),
                ("alias_name", ALIAS),
                ("prev_name", ALIAS),
            ):
                if field in col:
                    for surface in _split_list(row[col[field]]):
                        yield LexiconEntry(surface, entity_id, symbol, priority)


def _ctd_header(handle) -> list[str]:
    previous = ""
    for line in handle:
        if not line.startswith("#"):
            break
        if previous.strip() == "# Fields:":
            return line.lstrip("#").strip().split("\t")
        previous = line
    raise LexiconFormatError("CTD file has no '# Fields:' header")


def iter_ctd(path: Path, id_field: str, name_field: str) -> Iterator[LexiconEntry]:
    with _open_text(path) as handle:
        header = _ctd_header(handle)
        col = {name: i for i, name in enumerate(header)}
        if not {id_field, name_field, "Synonyms"} <= col.keys():
            raise LexiconFormatError(f"CTD file lacks {id_field}/{name_field}/Synonyms columns")
        for line in handle:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            row += [""] * (len(header) - len(row))
            name, entity_id = row[col[name_field]], row[col[id_field]]
            yield LexiconEntry(name, entity_id, name, PREFERRED)
            for surface in _split_list(row[col["Synonyms"]]):
                yield LexiconEntry(surface, entity_id, name, SYNONYM)


def build_index(entries: Iterable[LexiconEntry], wanted: set[str] | None = None) -> dict[str, tuple[str, str]]:
    """Map normalisation key -> (entity_id, canonical name); lowest priority wins, then first seen.
    Each surface is indexed under its exact key and (penalised) singularised key."""
    best: dict[str, tuple[int, str, str]] = {}
    for entry in entries:
        for rank, key in enumerate(lookup_keys(entry.surface)):
            if not key or (wanted is not None and key not in wanted):
                continue
            priority = entry.priority + rank * _SINGULAR_PENALTY
            current = best.get(key)
            if current is None or priority < current[0]:
                best[key] = (priority, entry.entity_id, entry.name)
    return {key: (eid, name) for key, (_, eid, name) in best.items()}


def lexicon_sources(lexicon_dir: Path = config.LEXICON_DIR) -> dict[str, Iterable[LexiconEntry]]:
    """Lazily-iterated vocabulary per entity type (re-iterable via fresh generators)."""
    lexicon_dir = Path(lexicon_dir)
    missing = [f for f in (HGNC_FILE, CTD_DISEASES_FILE, CTD_CHEMICALS_FILE) if not (lexicon_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing vocabularies {missing}; run `bioscrolls download-lexicons`")
    return {
        config.GENE: iter_hgnc(lexicon_dir / HGNC_FILE),
        config.DISEASE: iter_ctd(lexicon_dir / CTD_DISEASES_FILE, "DiseaseID", "DiseaseName"),
        config.CHEMICAL: iter_ctd(lexicon_dir / CTD_CHEMICALS_FILE, "ChemicalID", "ChemicalName"),
    }


def download_lexicons(lexicon_dir: Path = config.LEXICON_DIR, session: requests.Session | None = None) -> list[Path]:
    """Fetch HGNC + CTD vocabularies (~30 MB total). Skips files already present."""
    http = session or requests.Session()
    lexicon_dir = Path(lexicon_dir)
    lexicon_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url, name in (
        (config.HGNC_URL, HGNC_FILE),
        (config.CTD_DISEASES_URL, CTD_DISEASES_FILE),
        (config.CTD_CHEMICALS_URL, CTD_CHEMICALS_FILE),
    ):
        path = lexicon_dir / name
        if not path.exists():
            response = http.get(url, timeout=config.HTTP_TIMEOUT_S * 4)
            if response.status_code != 200:
                raise RuntimeError(f"download of {url} failed: HTTP {response.status_code}")
            path.write_bytes(response.content)
            log.info("downloaded %s", path)
        paths.append(path)
    return paths
