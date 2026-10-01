"""Shared plumbing for the fine-tuning scripts (device, seeding, optimisation loop)."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from bioscrolls.config import TrainConfig

log = logging.getLogger(__name__)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shuffled_batches(items: Sequence[Any], batch_size: int, rng: random.Random) -> Iterator[list[Any]]:
    order = list(range(len(items)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[i] for i in order[start : start + batch_size]]


def ordered_batches(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def build_optimizer(model: nn.Module, cfg: TrainConfig, total_steps: int):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    return optimizer, scheduler


def fit(
    model: nn.Module,
    train_items: Sequence[Any],
    cfg: TrainConfig,
    loss_fn: Callable[[nn.Module, list[Any]], torch.Tensor],
    evaluate: Callable[[nn.Module], float],
    save_best: Callable[[nn.Module], None],
) -> dict[str, Any]:
    """Train for ``cfg.epochs``; after each epoch score on dev and keep the best."""
    rng = random.Random(cfg.seed)
    steps_per_epoch = (len(train_items) + cfg.batch_size - 1) // cfg.batch_size
    optimizer, scheduler = build_optimizer(model, cfg, steps_per_epoch * cfg.epochs)
    history, best_score, best_epoch = [], -1.0, -1
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(shuffled_batches(train_items, cfg.batch_size, rng), start=1):
            loss = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, steps_per_epoch, running / step)
        model.eval()
        score = evaluate(model)
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch, "dev_score": score})
        log.info("epoch %d dev score %.4f (%.0fs elapsed)", epoch, score, time.time() - started)
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_best(model)
    return {
        "history": history,
        "best_dev_score": best_score,
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - started, 1),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
