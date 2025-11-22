"""Optimizer and scheduler utilities."""

from __future__ import annotations

import torch

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import Optimizer

from ai_pipeline.config.schema import TrainingConfig


def create_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> Optimizer:
    """Create optimizer from training config."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_scheduler(
    optimizer: Optimizer,
    cfg: TrainingConfig,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create LR scheduler from training config."""
    warmup_steps = cfg.warmup_steps
    if warmup_steps == 0 and cfg.warmup_ratio > 0.0:
        warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    if cfg.lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if cfg.lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def lambda_fn(step: int) -> float:
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
