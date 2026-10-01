"""``bioscrolls`` command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from bioscrolls import config
from bioscrolls.graph.store import open_store

log = logging.getLogger("bioscrolls")

db_option = click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=config.DEFAULT_DB_PATH,
    show_default=True,
    help="SQLite database file.",
)
year_type = click.IntRange(1800, 2100)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    """BioScrolls: mine PubMed for neurological gene-disease-drug associations."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("urllib3", "filelock", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------- data + training


@main.command("download-corpus")
def download_corpus() -> None:
    """Download the BioRED corpus used for fine-tuning and evaluation."""
    from bioscrolls.corpora.biored import download_biored

    click.echo(f"BioRED ready at {download_biored()}")


@main.command("download-lexicons")
def download_lexicons_cmd() -> None:
    """Download HGNC + CTD vocabularies used for entity normalisation."""
    from bioscrolls.normalize.lexicon import download_lexicons

    for path in download_lexicons():
        click.echo(f"ready: {path}")


@main.command("train-ner")
@click.option("--epochs", type=click.IntRange(1, 50), default=None, help="Override configured epochs.")
def train_ner_cmd(epochs: int | None) -> None:
    """Fine-tune BioBERT for Gene/Disease/Chemical NER on BioRED."""
    from dataclasses import replace

    from bioscrolls.ner.train import train_ner

    cfg = replace(config.NER_TRAIN, epochs=epochs) if epochs else config.NER_TRAIN
    results = train_ner(cfg=cfg)
    click.echo(json.dumps(results["test"]["micro"], indent=2))


@main.command("train-re")
@click.option("--epochs", type=click.IntRange(1, 50), default=None, help="Override configured epochs.")
def train_re_cmd(epochs: int | None) -> None:
    """Fine-tune the BioBERT relation classifier on BioRED and evaluate vs co-occurrence."""
    from dataclasses import replace

    from bioscrolls.relation.train import train_re

    cfg = replace(config.RE_TRAIN, epochs=epochs) if epochs else config.RE_TRAIN
    results = train_re(cfg=cfg)
    click.echo(json.dumps(results["test"]["model"]["sentence"]["typed"], indent=2))


# ---------------------------------------------------------------------------- pipeline


@main.command()
@db_option
@click.option("--start", type=year_type, default=config.DEFAULT_YEAR_START, show_default=True)
@click.option("--end", type=year_type, default=config.DEFAULT_YEAR_END, show_default=True)
@click.option(
    "--per-year",
    type=click.IntRange(1, 500),
    default=config.DEFAULT_PER_YEAR,
    show_default=True,
    help="Abstracts per topic per year.",
)
@click.option(
    "--non-english",
    type=click.IntRange(0, 500),
    default=config.DEFAULT_NON_ENGLISH_PER_YEAR,
    show_default=True,
    help="Extra non-English-language articles per topic per year.",
)
@click.option(
    "--topic",
    "topics",
    multiple=True,
    type=click.Choice(sorted(config.NEURO_QUERIES)),
    help="Restrict to these topics (default: all).",
)
def ingest(db_path: Path, start: int, end: int, per_year: int, non_english: int, topics: tuple[str, ...]) -> None:
    """Fetch PubMed records via NCBI E-utilities (cached, rate limited)."""
    from bioscrolls.ingest.eutils import EutilsClient
    from bioscrolls.ingest.pipeline import plan_queries, run_ingest

    queries = {k: v for k, v in config.NEURO_QUERIES.items() if not topics or k in topics}
    specs = plan_queries(queries, start, end, per_year, non_english)
    click.echo(f"running {len(specs)} searches...")
    result = run_ingest(EutilsClient(), specs)
    with open_store(db_path) as store:
        stored = store.upsert_documents(result.documents)
        store.record_search_counts(result.search_counts)
    non_en = sum(not d.is_english for d in result.documents)
    click.echo(f"stored {stored} documents ({non_en} non-English-language articles) in {db_path}")


@main.command()
@db_option
@click.option("--batch-docs", type=click.IntRange(1, 5000), default=400, show_default=True)
@click.option("--limit", type=click.IntRange(1), default=None, help="Process at most N documents.")
@click.option("--redo", is_flag=True, help="Re-extract documents that were already processed.")
def extract(db_path: Path, batch_docs: int, limit: int | None, redo: bool) -> None:
    """Run NER, normalisation and relation classification over ingested documents."""
    from bioscrolls.extraction import extract_documents, storage_rows
    from bioscrolls.ner.predict import NerTagger
    from bioscrolls.normalize.lexicon import lexicon_sources
    from bioscrolls.normalize.normalizer import Normalizer, corpus_abbreviations
    from bioscrolls.relation.predict import RelationClassifier
    from bioscrolls.training_utils import pick_device

    device = pick_device()
    tagger = NerTagger.from_dir(config.NER_MODEL_DIR, device)
    classifier = RelationClassifier.from_dir(config.RE_MODEL_DIR, device)
    lexicon_sources()  # fail fast if vocabularies are missing

    def factory(requests):
        return Normalizer.build(lexicon_sources(), requests)

    with open_store(db_path) as store:
        all_docs = store.documents()
        abbreviations = corpus_abbreviations(d.text for d in all_docs)
        todo = all_docs if redo else store.documents(only_unextracted=True)
        todo = todo[:limit] if limit else todo
        click.echo(f"extracting {len(todo)} documents on {device}")
        for start in range(0, len(todo), batch_docs):
            batch = todo[start : start + batch_docs]
            for result in extract_documents(batch, tagger, classifier, factory, abbreviations):
                store.save_extraction(result.pmid, result.scope, *storage_rows(result))
            store.conn.commit()
            click.echo(f"  {min(start + batch_docs, len(todo))}/{len(todo)}")
    click.echo("done")


@main.command("build-graph")
@db_option
@click.option(
    "--min-confidence",
    type=click.FloatRange(0.0, 1.0),
    default=config.MIN_RELATION_CONFIDENCE,
    show_default=True,
    help="Minimum P(relation) for a sentence to count as evidence.",
)
def build_graph_cmd(db_path: Path, min_confidence: float) -> None:
    """Aggregate extractions into the time-indexed knowledge graph."""
    from bioscrolls.graph.build import build_graph

    with open_store(db_path) as store:
        sizes = build_graph(store, min_confidence)
    for table, size in sizes.items():
        click.echo(f"{table:>13}: {size}")


from bioscrolls.cli_read import register_read_commands  # noqa: E402

register_read_commands(main, db_option)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
