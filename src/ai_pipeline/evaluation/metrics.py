"""Evaluation metrics."""

from __future__ import annotations

import math

from typing import Dict


def compute_perplexity(metrics: Dict[str, float], loss_key: str = "eval_loss") -> float:
    """Compute perplexity from loss metric."""
    loss = metrics.get(loss_key)
    if loss is None:
        raise KeyError(f"'{loss_key}' not found in metrics dict")

    return float(math.exp(loss))
