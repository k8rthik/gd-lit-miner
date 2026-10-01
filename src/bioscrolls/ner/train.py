"""Fine-tune BioBERT for Gene/Disease/Chemical NER on BioRED.

Train on the BioRED train split, select the epoch with the best dev micro-F1,
then report strict span-level P/R/F1 on the held-out test split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bioscrolls import config
from bioscrolls.corpora.biored import load_split
from bioscrolls.ner.encoding import IGNORE_INDEX, spans_to_token_tags
from bioscrolls.ner.evaluate import span_metrics
from bioscrolls.ner.examples import NerExample, ner_examples
from bioscrolls.ner.predict import NerTagger
from bioscrolls.training_utils import fit, pick_device, set_seed, write_json

log = logging.getLogger(__name__)


def _encode(tokenizer, examples: list[NerExample], max_length: int) -> list[dict]:
    encoded = []
    for ex in examples:
        enc = tokenizer(ex.text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        labels = spans_to_token_tags([tuple(o) for o in enc["offset_mapping"]], ex.spans)
        encoded.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})
    return encoded


def _collate(batch: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    width = max(len(item["input_ids"]) for item in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        rows = [item[key] + [value] * (width - len(item[key])) for item in batch]
        return torch.tensor(rows, dtype=torch.long, device=device)

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "labels": pad("labels", IGNORE_INDEX),
    }


def _score(tagger: NerTagger, examples: list[NerExample]) -> dict:
    predicted = tagger.tag([ex.text for ex in examples])
    return span_metrics([list(ex.spans) for ex in examples], predicted)


def _serialise(metrics: dict) -> dict:
    return {
        "micro": metrics["micro"].as_dict(),
        "per_type": {k: v.as_dict() for k, v in metrics["per_type"].items()},
    }


def train_ner(output_dir: Path = config.NER_MODEL_DIR, cfg: config.TrainConfig = config.NER_TRAIN) -> dict:
    set_seed(cfg.seed)
    device = pick_device()
    log.info("training NER on %s", device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, **config.BASE_TOKENIZER_KWARGS)
    id2label = dict(enumerate(config.NER_LABELS))
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(config.NER_LABELS),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    ).to(device)

    train, dev, test = (ner_examples(load_split(s)) for s in config.BIORED_SPLITS)
    train_items = _encode(tokenizer, train, cfg.max_length)

    def loss_fn(m, batch):
        return m(**_collate(batch, tokenizer.pad_token_id, device)).loss

    def evaluate(m):
        return _score(NerTagger(m, tokenizer, device, cfg.max_length), dev)["micro"].f1

    def save_best(m):
        m.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    summary = fit(model, train_items, cfg, loss_fn, evaluate, save_best)
    best = NerTagger.from_dir(output_dir, device)
    results = {
        "task": "ner",
        "corpus": "BioRED",
        "base_model": cfg.base_model,
        "device": str(device),
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "training": summary,
        "dev": _serialise(_score(best, dev)),
        "test": _serialise(_score(best, test)),
    }
    write_json(config.RESULTS_DIR / "ner_metrics.json", results)
    return results
