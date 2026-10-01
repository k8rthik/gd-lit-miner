"""SQLite persistence for documents, extractions and the derived knowledge graph."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from bioscrolls.models import Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    pmid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    year INTEGER,
    languages TEXT NOT NULL,
    journal TEXT,
    vernacular_title TEXT,
    other_abstracts TEXT,         -- JSON [[language, text], ...]: stored, not NLP-processed
    query_label TEXT,
    nlp_scope TEXT,               -- 'title+abstract' | 'title' | 'none' (set by extract)
    extracted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS search_counts (
    label TEXT, year INTEGER, kind TEXT, total INTEGER, retrieved INTEGER
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL, matched INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    pmid TEXT NOT NULL, sentence_index INTEGER NOT NULL, start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
    surface TEXT NOT NULL, type TEXT NOT NULL, entity_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relations (
    pmid TEXT NOT NULL, sentence_index INTEGER NOT NULL, head_id TEXT NOT NULL, tail_id TEXT NOT NULL,
    label TEXT NOT NULL, confidence REAL NOT NULL, sentence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mentions_pmid ON mentions(pmid);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_pmid ON relations(pmid);
CREATE TABLE IF NOT EXISTS edges (
    head_id TEXT, tail_id TEXT, label TEXT, n_docs INTEGER, n_sentences INTEGER,
    mean_confidence REAL, max_confidence REAL, combined_confidence REAL,
    first_year INTEGER, last_year INTEGER, PRIMARY KEY (head_id, tail_id, label)
);
CREATE TABLE IF NOT EXISTS edge_years (
    head_id TEXT, tail_id TEXT, label TEXT, year INTEGER, n_docs INTEGER, mean_confidence REAL
);
CREATE TABLE IF NOT EXISTS entity_stats (
    entity_id TEXT PRIMARY KEY, n_docs INTEGER, n_mentions INTEGER, first_year INTEGER, last_year INTEGER
);
CREATE TABLE IF NOT EXISTS entity_years (entity_id TEXT, year INTEGER, n_docs INTEGER);
CREATE TABLE IF NOT EXISTS year_docs (year INTEGER PRIMARY KEY, n_docs INTEGER);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

GRAPH_TABLES = ("edges", "edge_years", "entity_stats", "entity_years", "year_docs")


class Store:
    """Thin repository over a SQLite connection. Use :func:`open_store`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.conn = connection

    # ------------------------------------------------------------------ documents

    def upsert_documents(self, documents: Iterable[Document]) -> int:
        rows = [
            (
                d.pmid,
                d.title,
                d.abstract,
                d.year,
                ",".join(d.languages),
                d.journal,
                d.vernacular_title,
                json.dumps([list(item) for item in d.other_abstracts], ensure_ascii=False),
                d.query_label,
            )
            for d in documents
        ]
        self.conn.executemany(
            """INSERT INTO documents (pmid, title, abstract, year, languages, journal, vernacular_title,
                   other_abstracts, query_label)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pmid) DO UPDATE SET title=excluded.title, abstract=excluded.abstract,
                   year=excluded.year, languages=excluded.languages, journal=excluded.journal,
                   vernacular_title=excluded.vernacular_title,
                   other_abstracts=excluded.other_abstracts""",
            rows,
        )
        return len(rows)

    def record_search_counts(self, counts: Iterable[tuple]) -> None:
        self.conn.executemany("INSERT INTO search_counts VALUES (?, ?, ?, ?, ?)", list(counts))

    def documents(self, only_unextracted: bool = False, limit: int | None = None) -> list[Document]:
        sql = "SELECT pmid, title, abstract, year, languages, journal, vernacular_title, query_label FROM documents"
        if only_unextracted:
            sql += " WHERE extracted = 0"
        sql += " ORDER BY pmid"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return [
            Document(
                pmid=r[0],
                title=r[1],
                abstract=r[2],
                year=r[3],
                languages=tuple(r[4].split(",")),
                journal=r[5] or "",
                vernacular_title=r[6] or "",
                query_label=r[7] or "",
            )
            for r in self.conn.execute(sql, params)
        ]

    # ------------------------------------------------------------------ extractions

    def save_extraction(
        self,
        pmid: str,
        scope: str,
        mentions: Sequence[tuple],
        relations: Sequence[tuple],
        entities: Iterable[tuple[str, str, str, bool]],
    ) -> None:
        """Replace a document's extractions atomically (idempotent re-runs)."""
        self.conn.execute("DELETE FROM mentions WHERE pmid = ?", (pmid,))
        self.conn.execute("DELETE FROM relations WHERE pmid = ?", (pmid,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO entities (entity_id, type, name, matched) VALUES (?, ?, ?, ?)",
            [(eid, etype, name, int(matched)) for eid, etype, name, matched in entities],
        )
        self.conn.executemany("INSERT INTO mentions VALUES (?, ?, ?, ?, ?, ?, ?)", mentions)
        self.conn.executemany("INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?)", relations)
        self.conn.execute("UPDATE documents SET extracted = 1, nlp_scope = ? WHERE pmid = ?", (scope, pmid))

    # ------------------------------------------------------------------ misc

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def clear_graph(self) -> None:
        for table in GRAPH_TABLES:
            self.conn.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names

    def count(self, table: str) -> int:
        if table not in {"documents", "mentions", "relations", "entities", *GRAPH_TABLES}:
            raise ValueError(f"unknown table {table!r}")
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608


@contextmanager
def open_store(path: Path) -> Iterator[Store]:
    """Open (creating if needed) the database; commit on success, roll back on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        yield Store(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
