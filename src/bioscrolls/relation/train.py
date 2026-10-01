"""Fine-tune a BioBERT relation classifier on BioRED sentence-level candidates.

Selects the epoch with the best dev typed micro-F1, then reports held-out test
metrics for the model and for a co-occurrence baseline, both at the
sentence-candidate level and aggregated to document-level entity pairs.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from bioscrolls import config
from bioscrolls.corpora.biored import AnnotatedDoc, load_split
from bioscrolls.relation.evaluate import cooccurrence_baseline, document_level_metrics, sentence_level_metrics
from bioscrolls.relation.examples import RelationExample, relation_examples
from bioscrolls.relation.model import MarkerRelationClassifier, build_tokenizer
from bioscrolls.relation.predict import RelationClassifier
from bioscrolls.training_utils import fit, pick_device, set_seed, write_json

log = logging.getLogger(__name__)
LABELS = config.RELATION_LABELS
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def _serialise(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if hasattr(value, "as_dict"):
            out[key] = value.as_dict()
        elif isinstance(value, dict):
            out[key] = {k: v.as_dict() for k, v in value.items()}
        else:
            out[key] = value
    return out


def _evaluate(predictions, examples: list[RelationExample], docs: list[AnnotatedDoc]) -> dict:
    pairs = [(label, conf) for label, conf, *_ in predictions]
    return {
        "sentence": _serialise(sentence_level_metrics([e.label for e in examples], [p for p, _ in pairs])),
        "document": _serialise(document_level_metrics(examples, pairs, docs)),
    }


def train_re(output_dir: Path = config.RE_MODEL_DIR, cfg: config.TrainConfig = config.RE_TRAIN) -> dict:
    set_seed(cfg.seed)
    device = pick_device()
    log.info("training relation classifier on %s", device)
    tokenizer = build_tokenizer(cfg.base_model)
    encoder = AutoModel.from_pretrained(cfg.base_model)
    encoder.resize_token_embeddings(len(tokenizer))
    head_id, tail_id = tokenizer.convert_tokens_to_ids([config.HEAD_START, config.TAIL_START])
    model = MarkerRelationClassifier(encoder, len(LABELS), head_id, tail_id).to(device)

    docs = {s: load_split(s) for s in config.BIORED_SPLITS}
    examples = {s: relation_examples(d) for s, d in docs.items()}
    train = examples["Train"]
    log.info("label distribution (train): %s", Counter(e.label for e in train))
    loss = nn.CrossEntropyLoss()

    def loss_fn(m, batch: list[RelationExample]):
        enc = tokenizer(
            [e.text for e in batch], truncation=True, max_length=cfg.max_length, padding=True, return_tensors="pt"
        )
        logits = m(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        target = torch.tensor([LABEL_TO_ID[e.label] for e in batch], device=device)
        return loss(logits, target)

    def evaluate(m):
        clf = RelationClassifier(m, tokenizer, LABELS, device, cfg.max_length)
        preds = clf.classify([e.text for e in examples["Dev"]])
        return sentence_level_metrics([e.label for e in examples["Dev"]], [p[0] for p in preds])["typed"].f1

    def save_best(m):
        m.save(output_dir, tokenizer, LABELS)

    summary = fit(model, train, cfg, loss_fn, evaluate, save_best)
    best = RelationClassifier.from_dir(output_dir, device)
    results = {
        "task": "relation_extraction",
        "corpus": "BioRED (Gene/Chemical/Disease cross-type pairs, sentence-level candidates)",
        "base_model": cfg.base_model,
        "device": str(device),
        "labels": list(LABELS),
        "sizes": {s: len(e) for s, e in examples.items()},
        "label_distribution": {s: dict(Counter(e.label for e in ex)) for s, ex in examples.items()},
        "training": summary,
    }
    for split in ("Dev", "Test"):
        ex, dd = examples[split], docs[split]
        model_preds = best.classify([e.text for e in ex])
        baseline_preds = [(label, conf, ()) for label, conf in cooccurrence_baseline(ex)]
        results[split.lower()] = {
            "model": _evaluate(model_preds, ex, dd),
            "cooccurrence_baseline": _evaluate(baseline_preds, ex, dd),
        }
    write_json(config.RESULTS_DIR / "re_metrics.json", results)
    return results
