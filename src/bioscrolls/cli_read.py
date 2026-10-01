"""Read-only CLI commands: query, top, trends, export, stats."""

from __future__ import annotations

from pathlib import Path

import click

from bioscrolls import config
from bioscrolls.graph.export import FORMATS, export_graph
from bioscrolls.graph.query import evidence_sentences, find_entities, neighbors, top_entities
from bioscrolls.graph.store import open_store
from bioscrolls.graph.trends import edge_trends, entity_series

year_type = click.IntRange(1800, 2100)


def _require_db(db_path: Path) -> None:
    if not db_path.exists():
        raise click.ClickException(f"{db_path} does not exist; run `bioscrolls ingest` first")


def _sparkline(series: tuple[tuple[int, int], ...]) -> str:
    return " ".join(f"{year}:{n}" for year, n in series)


def register_read_commands(main: click.Group, db_option) -> None:
    @main.command()
    @db_option
    @click.option("--entity", required=True, help="Entity name or id, e.g. APOE or MESH:D000544.")
    @click.option("--since", type=year_type, default=None)
    @click.option("--until", type=year_type, default=None)
    @click.option("--label", type=click.Choice(config.RELATION_LABELS[1:]), default=None)
    @click.option("--min-docs", type=click.IntRange(1), default=1, show_default=True)
    @click.option("--limit", type=click.IntRange(1, 500), default=config.DEFAULT_QUERY_LIMIT, show_default=True)
    @click.option(
        "--evidence",
        "n_evidence",
        type=click.IntRange(0, 20),
        default=0,
        help="Show N supporting sentences per association.",
    )
    def query(db_path, entity, since, until, label, min_docs, limit, n_evidence) -> None:
        """Associations of an entity, optionally within a publication-year window."""
        _require_db(db_path)
        with open_store(db_path) as store:
            matches = find_entities(store, entity)
            if not matches:
                raise click.ClickException(f"no entity matching {entity!r}")
            target = matches[0]
            click.echo(f"{target.name} [{target.type}, {target.entity_id}] in {target.n_docs} docs")
            if len(matches) > 1:
                click.echo("  other matches: " + ", ".join(f"{m.name} ({m.entity_id})" for m in matches[1:5]))
            rows = neighbors(store, target.entity_id, since, until, label, min_docs, limit)
            if not rows:
                click.echo("no associations in this window")
            click.echo(f"{'partner':<32} {'type':<9} {'relation':<21} {'docs':>4} {'conf':>5}  years")
            for r in rows:
                click.echo(
                    f"{r.neighbor_name[:32]:<32} {r.neighbor_type:<9} {r.label:<21} {r.n_docs:>4} "
                    f"{r.mean_confidence:>5.2f}  {r.first_year}-{r.last_year}"
                )
                for pmid, year, lab, conf, sentence in (
                    evidence_sentences(store, target.entity_id, r.neighbor_id, since, n_evidence) if n_evidence else []
                ):
                    click.echo(f"    PMID {pmid} ({year}) {lab} {conf:.2f}: {sentence[:220]}")

    @main.command()
    @db_option
    @click.option("--type", "entity_type", type=click.Choice(config.ENTITY_TYPES), default=None)
    @click.option("--since", type=year_type, default=None)
    @click.option("--limit", type=click.IntRange(1, 500), default=config.DEFAULT_QUERY_LIMIT, show_default=True)
    def top(db_path, entity_type, since, limit) -> None:
        """Most frequently mentioned entities (document frequency)."""
        _require_db(db_path)
        with open_store(db_path) as store:
            for entity_id, name, etype, n in top_entities(store, entity_type, since, limit):
                click.echo(f"{n:>6}  {etype:<9} {name[:40]:<40} {entity_id}")

    @main.command()
    @db_option
    @click.option("--entity", default=None, help="Only associations of this entity.")
    @click.option("--recent-years", type=click.IntRange(1, 30), default=config.EMERGING_RECENT_YEARS, show_default=True)
    @click.option("--min-docs", type=click.IntRange(1), default=config.EMERGING_MIN_DOCS, show_default=True)
    @click.option("--limit", type=click.IntRange(1, 500), default=15, show_default=True)
    def trends(db_path, entity, recent_years, min_docs, limit) -> None:
        """Per-year counts and emerging associations (recent vs earlier report rate)."""
        _require_db(db_path)
        with open_store(db_path) as store:
            entity_id = None
            if entity:
                matches = find_entities(store, entity)
                if not matches:
                    raise click.ClickException(f"no entity matching {entity!r}")
                entity_id = matches[0].entity_id
                click.echo(f"{matches[0].name}: docs/year (per 1k processed docs)")
                click.echo("  " + "  ".join(f"{y}:{n}({r:.0f})" for y, n, r in entity_series(store, entity_id)))
            rows = edge_trends(store, entity_id, recent_years, min_docs)[:limit]
            click.echo(f"{'association':<58} {'early/1k':>8} {'recent/1k':>9} {'ratio':>6}  series")
            for t in rows:
                e = t.emergence
                pair = f"{t.head_name[:22]} -[{t.label[:12]}]- {t.tail_name[:20]}"
                click.echo(
                    f"{pair:<58} {e.early_rate:>8.1f} {e.recent_rate:>9.1f} {e.rate_ratio:>6.2f}  "
                    f"{_sparkline(t.series)}"
                )

    @main.command()
    @db_option
    @click.option("--format", "fmt", type=click.Choice(FORMATS), default="graphml", show_default=True)
    @click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
    @click.option("--min-docs", type=click.IntRange(1), default=1, show_default=True)
    @click.option("--since", type=year_type, default=None)
    def export(db_path, fmt, out, min_docs, since) -> None:
        """Export the graph as GraphML or node-link JSON."""
        _require_db(db_path)
        with open_store(db_path) as store:
            nodes, edges = export_graph(store, out, fmt, min_docs, since)
        click.echo(f"wrote {nodes} nodes / {edges} edges to {out}")

    @main.command()
    @db_option
    def stats(db_path) -> None:
        """Corpus coverage: languages, NLP scope, normalisation rate, graph size."""
        _require_db(db_path)
        with open_store(db_path) as store:
            q = store.conn.execute
            click.echo(f"documents: {store.count('documents')}")
            for row in q("SELECT languages, COUNT(*) FROM documents GROUP BY languages ORDER BY 2 DESC LIMIT 15"):
                click.echo(f"  language {row[0]:<12} {row[1]}")
            for row in q("SELECT COALESCE(nlp_scope, 'not extracted'), COUNT(*) FROM documents GROUP BY 1"):
                click.echo(f"  nlp scope {row[0]:<16} {row[1]}")
            total, matched = q(
                "SELECT COUNT(*), SUM(e.matched) FROM mentions m JOIN entities e ON e.entity_id = m.entity_id"
            ).fetchone()
            rate = (matched or 0) / total if total else 0.0
            click.echo(f"mentions: {total} ({rate:.1%} normalised to a vocabulary id)")
            for table in ("entities", "relations", "edges"):
                click.echo(f"{table}: {store.count(table)}")
