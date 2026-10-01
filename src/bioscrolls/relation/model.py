"""BioBERT relation classifier with entity-start pooling (Baldini Soares et al., 2019).

The sentence is wrapped with ``[E1]``/``[E2]`` markers; the classifier
concatenates the final hidden states at the two opening markers and applies a
linear layer. Falls back to ``[CLS]`` when a marker was truncated away.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from bioscrolls import config

_META_FILE = "relation_head.json"
_HEAD_WEIGHTS = "relation_head.pt"


def build_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, **config.BASE_TOKENIZER_KWARGS)
    tokenizer.add_special_tokens({"additional_special_tokens": list(config.ENTITY_MARKERS)})
    return tokenizer


class MarkerRelationClassifier(nn.Module):
    def __init__(self, encoder, num_labels: int, head_token_id: int, tail_token_id: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head_token_id = head_token_id
        self.tail_token_id = tail_token_id
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_labels)

    def _marker_states(self, hidden: torch.Tensor, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
        is_marker = (input_ids == token_id).int()
        position = is_marker.argmax(dim=1)  # 0 ([CLS]) when the marker is absent
        return hidden[torch.arange(hidden.size(0), device=hidden.device), position]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = torch.cat(
            [
                self._marker_states(hidden, input_ids, self.head_token_id),
                self._marker_states(hidden, input_ids, self.tail_token_id),
            ],
            dim=-1,
        )
        return self.classifier(self.dropout(pooled))

    def save(self, directory: Path, tokenizer, labels: tuple[str, ...]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / "encoder")
        tokenizer.save_pretrained(directory / "encoder")
        torch.save(self.classifier.state_dict(), directory / _HEAD_WEIGHTS)
        (directory / _META_FILE).write_text(json.dumps({"labels": list(labels)}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, device: torch.device):
        directory = Path(directory)
        meta_path = directory / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"no relation model at {directory}; run `bioscrolls train-re` first")
        labels = tuple(json.loads(meta_path.read_text(encoding="utf-8"))["labels"])
        tokenizer = AutoTokenizer.from_pretrained(directory / "encoder")
        encoder = AutoModel.from_pretrained(directory / "encoder")
        head_id, tail_id = tokenizer.convert_tokens_to_ids([config.HEAD_START, config.TAIL_START])
        model = cls(encoder, len(labels), head_id, tail_id)
        model.classifier.load_state_dict(torch.load(directory / _HEAD_WEIGHTS, map_location="cpu"))
        return model.to(device).eval(), tokenizer, labels
