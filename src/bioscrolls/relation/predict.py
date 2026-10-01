"""Batched relation classification over entity-marked sentences."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from bioscrolls import config
from bioscrolls.relation.model import MarkerRelationClassifier

Prediction = tuple[str, float, tuple[tuple[str, float], ...]]


class RelationClassifier:
    def __init__(
        self,
        model,
        tokenizer,
        labels: tuple[str, ...],
        device: torch.device,
        max_length: int = config.RE_TRAIN.max_length,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._labels = labels
        self._device = device
        self._max_length = max_length

    @classmethod
    def from_dir(cls, model_dir: Path, device: torch.device) -> RelationClassifier:
        model, tokenizer, labels = MarkerRelationClassifier.load(model_dir, device)
        return cls(model, tokenizer, labels, device)

    @torch.no_grad()
    def classify(self, marked_texts: Sequence[str], batch_size: int = config.INFERENCE_BATCH_SIZE) -> list[Prediction]:
        out: list[Prediction] = []
        for start in range(0, len(marked_texts), batch_size):
            batch = list(marked_texts[start : start + batch_size])
            enc = self._tokenizer(
                batch, truncation=True, max_length=self._max_length, padding=True, return_tensors="pt"
            )
            logits = self._model(enc["input_ids"].to(self._device), enc["attention_mask"].to(self._device))
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                best = max(range(len(row)), key=row.__getitem__)
                out.append((self._labels[best], row[best], tuple(zip(self._labels, row, strict=True))))
        return out
