# BioScrolls

A working pipeline that mines PubMed abstracts about neurological disease for
**gene–disease, drug–disease and gene–drug associations**, and turns them into a
**time-indexed knowledge graph** you can query by entity and publication year.

The relation extractor is a BioBERT encoder fine-tuned here, on
[BioRED](https://academic.oup.com/bib/article/23/5/bbac282/6645993); so is the
entity tagger. Everything is measured on BioRED's held-out test split and
compared with a co-occurrence baseline, and all the numbers below were produced
by the commands in this README on a 2023 MacBook Pro (M3 Pro, 18 GB, MPS).

```
PubMed (E-utilities)  ->  sentences  ->  BioBERT NER  ->  HGNC/CTD normalisation
                                                             |
                      SQLite knowledge graph  <-  BioBERT relation classifier
                                                             |
                                  query / trends / GraphML-JSON export
```

## Quick start

```bash
uv sync                               # Python 3.12, torch + transformers
uv run bioscrolls download-corpus     # BioRED (2 MB)
uv run bioscrolls download-lexicons   # HGNC + CTD vocabularies (28 MB)

uv run bioscrolls train-ner           # ~13 min on MPS
uv run bioscrolls train-re            # ~21 min on MPS

uv run bioscrolls ingest              # ~3,000 abstracts, 2005-2024 (cached)
uv run bioscrolls extract             # NER + normalisation + relations
uv run bioscrolls build-graph

uv run bioscrolls query --entity APOE --since 2015
uv run bioscrolls trends --entity "Parkinson Disease"
uv run bioscrolls export --format graphml --out exports/kg.graphml --min-docs 2
uv run bioscrolls evaluate-pipeline   # end-to-end numbers on BioRED test
```

Nothing downloaded or produced is committed: `data/`, `models/`, `logs/` and
`exports/` are git-ignored. A full rebuild from an empty checkout takes
about 50 minutes: 13 min NER fine-tuning, 21 min relation fine-tuning, ~7 min
ingestion (first run; cached afterwards) and ~8 min extraction.

### Commands

| Command | What it does |
| --- | --- |
| `ingest` | Year-stratified PubMed search + fetch for five neurological topics |
| `extract` | Sentence split, NER, entity normalisation, relation classification |
| `build-graph` | Aggregate sentence predictions into typed, dated edges |
| `query --entity X [--since/--until/--label/--evidence N]` | Associations of an entity, with supporting sentences |
| `top [--type Gene]` | Entity frequency (document counts) |
| `trends [--entity X]` | Per-year counts and emerging associations |
| `export --format graphml\|json` | Graph for Gephi/Cytoscape or D3 |
| `stats` | Corpus coverage: languages, NLP scope, normalisation rate |
| `train-ner`, `train-re`, `evaluate-pipeline` | Fine-tuning and measurement |

`bioscrolls --help` and `bioscrolls COMMAND --help` list every option.

## Data provenance

| Source | Used for | Licence / terms |
| --- | --- | --- |
| [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | Abstract corpus (titles, abstracts, languages, dates) | NCBI usage policy; ≤3 req/s honoured, responses cached under `data/cache/` |
| [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) (Luo et al., 2022) | NER + relation training/eval; 600 abstracts, 400/100/100 | Public NCBI release |
| [HGNC complete set](https://www.genenames.org/download/archive/) | Gene normalisation to NCBI Gene IDs | CC0 |
| [CTD](https://ctdbase.org/downloads/) diseases + chemicals | Disease/chemical normalisation to MeSH IDs | CTD terms of use (academic use free) |
| `dmis-lab/biobert-base-cased-v1.2` | Encoder for both fine-tuned models | Apache-2.0 |

The corpus: five MeSH queries (Alzheimer disease, Parkinson disease, ALS,
epilepsy, multiple sclerosis), **30 abstracts per topic per year for 2005–2024**
plus **3 non-English-language articles per topic per year**, restricted to
records that have an abstract. That gives **3,033 unique abstracts**
(≈150 per year, so year-over-year counts are comparable rather than tracking
PubMed's own growth). Every E-utilities response is cached by request
parameters, so re-running `ingest` is free and reproducible.

## Measured results

### Entity recognition (BioRED test, strict exact-span match)

Fine-tuned `dmis-lab/biobert-base-cased-v1.2`, 5 epochs, best dev epoch 4,
12.7 min on MPS. Sentence-level, three types.

| Type | P | R | F1 |
| --- | --- | --- | --- |
| Gene | 0.853 | 0.889 | **0.871** |
| Disease | 0.770 | 0.840 | **0.803** |
| Chemical | 0.812 | 0.905 | **0.856** |
| **micro** | **0.815** | **0.877** | **0.845** |

(dev micro F1 0.853.)

### Relation extraction (BioRED test)

Candidates are in-scope entity pairs co-occurring in a sentence, labelled from
BioRED's document-level annotations; `[E1]`/`[E2]` markers with entity-start
pooling. Four labels: `None`, `Association`, `Positive_Correlation`,
`Negative_Correlation` (BioRED's five rare types are folded into
`Association`). The baseline predicts `Association` for **every** co-occurring
pair — i.e. plain co-occurrence.

Sentence-candidate level, 1,387 candidates:

| | binary P | binary R | binary F1 | typed P | typed R | typed F1 | macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Co-occurrence baseline | 0.697 | **1.000** | 0.821 | 0.384 | 0.551 | 0.452 | 0.185 |
| Fine-tuned BioBERT | **0.815** | 0.878 | **0.845** | **0.631** | **0.680** | **0.655** | **0.631** |

Per label (model vs. baseline, which can only ever predict `Association`):

| Label | support | model F1 | baseline F1 |
| --- | --- | --- | --- |
| `Association` | 532 | 0.686 | 0.555 |
| `Positive_Correlation` | 268 | 0.640 | 0.000 |
| `Negative_Correlation` | 166 | 0.565 | 0.000 |

Document level (unordered identifier pair per abstract, most confident sentence
wins), against **all** 632 gold relations:

| | binary P | binary R | binary F1 | typed F1 |
| --- | --- | --- | --- | --- |
| Co-occurrence baseline | 0.620 | 0.853 | 0.718 | 0.369 |
| Fine-tuned BioBERT | **0.758** | 0.737 | **0.747** | **0.566** |

So the fine-tuning buys **+20 F1 points of typed accuracy over co-occurrence**
(0.655 vs 0.452 at candidate level, 0.566 vs 0.369 at document level) and turns
a 0.70-precision firehose into a 0.82-precision extractor. It cannot buy
recall: 14.7% of gold relations are never expressed within a single sentence,
so **sentence-level recall is capped at 0.853** on this split, and the baseline's
"perfect" binary recall is just that cap.

Best dev epoch 4 of 4 (dev typed F1 0.656), 21.3 min on MPS.

### Whole pipeline on BioRED test (predicted entities, no gold anything)

`evaluate-pipeline` runs the deployed path — predicted NER, dictionary
normalisation, relation classification — and scores the resulting
`(PMID, {id, id})` pairs against BioRED's gold relations.

| | P | R | F1 |
| --- | --- | --- | --- |
| Co-occurrence of predicted entities | 0.167 | 0.389 | 0.234 |
| Full pipeline | **0.223** | 0.358 | **0.275** |

This is much lower than the component scores, and the bottleneck is measurable:
**normalisation, given gold spans and gold types, is only 67% accurate.**

| Type | mentions | accuracy | hit a vocabulary entry |
| --- | --- | --- | --- |
| Gene | 1,178 | 0.543 | 0.750 |
| Disease | 916 | 0.674 | 0.704 |
| Chemical | 745 | 0.862 | 0.891 |
| **overall** | 2,839 | **0.669** | 0.772 |

Genes are the worst, and mostly not for the reason you would guess: **HGNC is
human-only, while 28.4% of BioRED test gene mentions have no human gold ID**
(mouse, rat, hamster orthologs). Of the 1,178 gene mentions, 294 miss the
vocabulary entirely and 244 match the right *symbol* with the wrong *species*
(e.g. `OPRM1` → human `4988` where the gold is Chinese hamster `100770962`).
Against the 0.716 ceiling a human-only vocabulary allows, 0.543 is 76% of
what was reachable. Fixing this needs species-aware normalisation (NCBI
`gene_info` plus organism detection), which is not implemented.

### The mined graph

From 3,033 abstracts, at the default evidence threshold P(relation) >= 0.5:

* **52,936 mentions** (17.5 per abstract), **8,785 distinct entities**, 2,515 of
  them (29%) carrying a vocabulary id; **68.5% of mentions** normalised.
* **18,304 classified candidate pairs** -> **8,921 edges** over 4,470 nodes:
  3,746 gene-disease, 3,417 chemical-disease, 1,758 gene-chemical.
  By label: 4,978 `Association`, 2,390 `Negative_Correlation`,
  1,553 `Positive_Correlation`.
* **589 edges have >= 2 supporting documents** — the long tail is single-abstract
  evidence, which is what `--min-docs` is for.

Spot checks come out where a neurologist would expect. Top genes by document
frequency: SOD1 (120), MAPT (96), TARDBP (56), SNCA (42), APP (32), C9orf72
(32), APOE (30). Top chemicals: anticonvulsants (97), levodopa (88), dopamine
(68), valproic acid (33). `query --entity APOE` returns
APOE–Alzheimer disease as its top edge (15 documents, mean confidence 0.94,
2005–2024), and `trends --min-docs 4` surfaces TREM2–Alzheimer disease and
natalizumab–multiple sclerosis among the fastest-rising associations, both of
which really did take off in the 2010s.

It also shows the failure modes honestly: `Met`, `D` and `anti` appear as
entities, and APOE picks up a spurious edge to `Nail-Patella Syndrome` from one
abstract.

## What "multilingual" actually covers

Being specific, because this is easy to overstate:

* The corpus deliberately includes **520 of 3,033 records (17%) whose article
  language is not English** — 25 languages, led by Japanese (106), Chinese (88),
  Russian (83), French (74), Polish (38), Spanish (36).
* PubMed supplies an **English title and English abstract** for essentially all
  of them (NLM translates the title; the publisher usually supplies an English
  abstract). The pipeline checks each abstract's language with a stopword
  heuristic and records the scope it actually used per document
  (`documents.nlp_scope`): on this corpus **3,032 of 3,033 documents were
  processed over title + abstract**, the one exception being an English record
  with an empty abstract.
* **116 records also carry a non-English `OtherAbstract`.** These are stored
  verbatim (`documents.other_abstracts`, JSON) but **are not run through the
  models.** Both models are English-only.
* So: BioScrolls **covers non-English-language literature** — it finds and
  indexes it, records its language, and mines the English text PubMed provides
  for it. It does **not** do NER or relation extraction on non-English text. A
  multilingual encoder (e.g. XLM-R, or mBERT-based biomedical NER) would be
  needed for that, and is not implemented here.

`bioscrolls stats` prints the language and scope breakdown for any database.

## Limitations

* **Sentence-level relations only.** Associations stated across sentences are
  invisible to the classifier; the measured ceiling on BioRED test is reported
  above.
* **Abstracts, not full text.** No PMC full-text ingestion.
* **Relation labels are coarse.** Four classes; BioRED's `Bind`,
  `Cotreatment`, `Comparison`, `Drug_Interaction` and `Conversion` (<2% of
  in-scope pairs) are folded into `Association`. Direction is not modelled —
  edges are stored head→tail by type order (gene → chemical → disease), not by
  causal direction.
* **No negation or speculation detection.** "X was not associated with Y" can
  still produce an edge; BioRED has no negation layer to train on.
* **Normalisation is dictionary-based and the largest single source of error**
  (67% accurate on BioRED test given gold spans — see above). It has no notion
  of context: an ambiguous symbol resolves to one HGNC entry regardless of
  organism, and HGNC covers only human genes.
  Mentions that miss the vocabulary keep a deterministic text-derived id
  (`gene-text:…`) so they still aggregate, and are flagged `matched = 0`.
* **Confidences are softmax outputs**, not calibrated probabilities. Edge
  `combined_confidence` is a noisy-OR over documents, which assumes independent
  evidence — optimistic when several abstracts copy the same claim.
* **Trends are about the literature, not biology.** A rising edge means the
  association is reported more often in this year-balanced sample, nothing more.
  With ~150 abstracts per year, per-edge yearly counts are small and noisy.
* **Topic coverage is five diseases**, ~150 abstracts per year. The query set is
  in `config.py`; it is a sample for building and measuring the pipeline, not a
  systematic review of the literature.
* **Entity noise is visible in the output.** Short or truncated spans (`Met`,
  `D`, `anti`) become nodes, and single-abstract edges include clear false
  positives. Filter with `--min-docs`.

## Layout

```
src/bioscrolls/
  config.py          paths, query set, label spaces, hyperparameters
  models.py          frozen Document / Span / Mention / RelationPrediction
  cli.py cli_read.py Click commands (write side / read side)
  runtime.py         loads trained models + vocabularies
  extraction.py      sentences -> NER -> normalise -> relations
  training_utils.py  device, seeding, optimiser, fit loop
  ingest/            eutils client (cache, rate limit, retry), XML parser, query plan
  text/              sentence splitter, offsets, English-text heuristic
  corpora/biored.py  PubTator reader + download
  ner/               BIO encoding, examples, train, predict, evaluate
  relation/          candidates/markers, examples, model, train, predict, evaluate
  normalize/         keys, abbreviations, HGNC/CTD readers, normalizer
  graph/             store (SQLite), build, metrics, query, trends, export
  evaluation/        P/R/F1, end-to-end scoring
tests/               pytest suite, network fully mocked
```

Tests: `uv run pytest`. Coverage of non-training code is **99%**
(`uv run pytest --cov=bioscrolls`); the two fine-tuning scripts and the shared
training loop are excluded, since they are exercised by the real runs above.
Lint: `uv run ruff check src tests`.

### Database schema

`documents`, `mentions`, `relations` hold the extraction record (one row per
candidate pair, with the marked sentence as evidence). `edges`, `edge_years`,
`entity_stats`, `entity_years`, `year_docs` are derived by `build-graph` and can
be rebuilt at any confidence threshold without re-running the models.

## Citation

BioRED: Luo L, Lai P-T, Wei C-H, Arighi CN, Lu Z. *BioRED: a rich biomedical
relation extraction dataset.* Briefings in Bioinformatics, 2022.
BioBERT: Lee J et al. *BioBERT: a pre-trained biomedical language
representation model for biomedical text mining.* Bioinformatics, 2020.
