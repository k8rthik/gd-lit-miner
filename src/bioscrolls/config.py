"""Central configuration: paths, query set, label spaces, and hyperparameters.

Everything tunable lives here so the rest of the code base has no magic values.
Paths can be redirected with the ``BIOSCROLLS_HOME`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# --------------------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("BIOSCROLLS_HOME", PROJECT_ROOT))
DATA_DIR = HOME / "data"
CACHE_DIR = DATA_DIR / "cache"
EUTILS_CACHE_DIR = CACHE_DIR / "eutils"
CORPORA_DIR = DATA_DIR / "corpora"
LEXICON_DIR = DATA_DIR / "lexicons"
MODELS_DIR = HOME / "models"
RESULTS_DIR = HOME / "results"
DEFAULT_DB_PATH = DATA_DIR / "bioscrolls.db"
NER_MODEL_DIR = MODELS_DIR / "ner-biored"
RE_MODEL_DIR = MODELS_DIR / "re-biored"

# --------------------------------------------------------------------------- NCBI E-utilities

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUTILS_TOOL_NAME = "bioscrolls"
NCBI_API_KEY_ENV = "NCBI_API_KEY"
NCBI_EMAIL_ENV = "NCBI_EMAIL"
# NCBI policy: <=3 requests/s without an API key, <=10 with one.
MIN_INTERVAL_NO_KEY_S = 0.4
MIN_INTERVAL_WITH_KEY_S = 0.11
HTTP_TIMEOUT_S = 30.0
HTTP_MAX_RETRIES = 4
HTTP_BACKOFF_BASE_S = 1.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
EFETCH_BATCH_SIZE = 200
ESEARCH_MAX_RETMAX = 9999

# Neurological query set: label -> PubMed query fragment (MeSH major-ish terms).
NEURO_QUERIES = MappingProxyType(
    {
        "alzheimer": '"Alzheimer Disease"[MeSH Terms]',
        "parkinson": '"Parkinson Disease"[MeSH Terms]',
        "als": '"Amyotrophic Lateral Sclerosis"[MeSH Terms]',
        "epilepsy": '"Epilepsy"[MeSH Terms]',
        "multiple_sclerosis": '"Multiple Sclerosis"[MeSH Terms]',
    }
)
DEFAULT_YEAR_START = 2005
DEFAULT_YEAR_END = 2024
DEFAULT_PER_YEAR = 30
DEFAULT_NON_ENGLISH_PER_YEAR = 3
ABSTRACT_FILTER = "hasabstract"
NON_ENGLISH_FILTER = "NOT english[Language]"

# --------------------------------------------------------------------------- corpora

BIORED_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
BIORED_SPLITS = ("Train", "Dev", "Test")
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
CTD_DISEASES_URL = "https://ctdbase.org/reports/CTD_diseases.tsv.gz"
CTD_CHEMICALS_URL = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"

# --------------------------------------------------------------------------- label spaces

GENE = "Gene"
DISEASE = "Disease"
CHEMICAL = "Chemical"
ENTITY_TYPES = (GENE, DISEASE, CHEMICAL)

# BioRED entity type -> our entity type. Types not listed are ignored.
BIORED_ENTITY_MAP = MappingProxyType(
    {
        "GeneOrGeneProduct": GENE,
        "DiseaseOrPhenotypicFeature": DISEASE,
        "ChemicalEntity": CHEMICAL,
    }
)

NER_LABELS = ("O",) + tuple(f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I"))

NO_RELATION = "None"
RELATION_LABELS = (
    NO_RELATION,
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
)
# BioRED relation type -> our (collapsed) relation label. The rare types (Bind,
# Cotreatment, Comparison, Drug_Interaction, Conversion: <2% of in-scope pairs)
# are folded into the generic "Association".
BIORED_RELATION_MAP = MappingProxyType(
    {
        "Association": "Association",
        "Positive_Correlation": "Positive_Correlation",
        "Negative_Correlation": "Negative_Correlation",
        "Bind": "Association",
        "Cotreatment": "Association",
        "Comparison": "Association",
        "Drug_Interaction": "Association",
        "Conversion": "Association",
    }
)
# Only cross-type pairs are in scope: gene-disease, chemical-disease, gene-chemical.
RELATION_PAIR_TYPES = frozenset(
    {
        frozenset({GENE, DISEASE}),
        frozenset({CHEMICAL, DISEASE}),
        frozenset({GENE, CHEMICAL}),
    }
)

# Entity markers inserted around the two candidate arguments for relation classification.
HEAD_START, HEAD_END = "[E1]", "[/E1]"
TAIL_START, TAIL_END = "[E2]", "[/E2]"
ENTITY_MARKERS = (HEAD_START, HEAD_END, TAIL_START, TAIL_END)

# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    weight_decay: float
    seed: int


BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"
# The BioBERT v1.2 checkpoint ships without a tokenizer config, so transformers
# defaults to lower-casing even though the vocabulary is cased. Force cased input.
BASE_TOKENIZER_KWARGS = MappingProxyType({"do_lower_case": False})
NER_TRAIN = TrainConfig(
    base_model=BASE_MODEL,
    max_length=192,
    batch_size=16,
    learning_rate=5e-5,
    epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=13,
)
RE_TRAIN = TrainConfig(
    base_model=BASE_MODEL,
    max_length=192,
    batch_size=16,
    learning_rate=3e-5,
    epochs=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=13,
)
INFERENCE_BATCH_SIZE = 32

# --------------------------------------------------------------------------- graph metrics

MIN_RELATION_CONFIDENCE = 0.5
EMERGING_RECENT_YEARS = 5
EMERGING_MIN_DOCS = 2
DEFAULT_QUERY_LIMIT = 25
