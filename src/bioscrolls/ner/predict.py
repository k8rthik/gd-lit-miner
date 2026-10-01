"""Batched BIO tagging with a fine-tuned token-classification model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bioscrolls import config
from bioscrolls.models import Span
from bioscrolls.ner.encoding import decode_tags, remove_overlaps

log = logging.getLogger(__name__)


class NerTagger:
    def __init__(self, model, tokenizer, device: torch.device, max_length: int = config.NER_TRAIN.max_length):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_dir(cls, model_dir: Path, device: torch.device) -> NerTagger:
        if not (Path(model_dir) / "config.json").exists():
            raise FileNotFoundError(f"no NER model at {model_dir}; run `bioscrolls train-ner` first")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
        return cls(model, tokenizer, device)

    @torch.no_grad()
    def tag(self, texts: Sequence[str], batch_size: int = config.INFERENCE_BATCH_SIZE) -> list[list[Span]]:
        results: list[list[Span]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self._tokenizer(
                batch,
                truncation=True,
                max_length=self._max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").tolist()
            inputs = {k: v.to(self._device) for k, v in encoded.items()}
            predictions = self._model(**inputs).logits.argmax(-1).cpu().tolist()
            for text, offs, preds, mask in zip(
                batch, offsets, predictions, encoded["attention_mask"].tolist(), strict=True
            ):
                length = sum(mask)
                spans = decode_tags([tuple(o) for o in offs[:length]], preds[:length], text)
                results.append(remove_overlaps(spans))
        return results
