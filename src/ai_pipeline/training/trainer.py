"""Training utilities."""

from __future__ import annotations

import torch
import math

from torch.utils.data import DataLoader
from typing import Optional

from ai_pipeline.training.optim import create_optimizer, create_scheduler
from ai_pipeline.data.dataset import JsonlTextDataset, collate_fn
from ai_pipeline.data.tokenization import create_tokenizer
from ai_pipeline.models.factory import create_causal_lm
from ai_pipeline.config.schema import FullConfig


def _get_device(cfg: FullConfig) -> torch.device:
    """Resolve device from run config."""
    if cfg.run.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.run.device)


def _get_autocast_dtype(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision.lower() in {"fp16", "float16"}:
        return torch.float16
    if mixed_precision.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


class Trainer:
    """Trainer for causal LM."""
    def __init__(self, cfg: FullConfig) -> None:
        self.cfg = cfg
        self.device = _get_device(cfg)

        self.output_dir = cfg.training.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = create_tokenizer(cfg.model)
        self.model = create_causal_lm(cfg.model)
        self.model.to(self.device)

        self.train_dataset = JsonlTextDataset(cfg.data)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.training.train_batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, cfg.data),
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / cfg.training.gradient_accumulation_steps
        )
        self.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch

        self.optimizer = create_optimizer(self.model, cfg.training)
        self.lr_scheduler = create_scheduler(
            self.optimizer,
            cfg.training,
            num_training_steps=self.max_train_steps,
        )

        self.autocast_dtype = _get_autocast_dtype(cfg.training.mixed_precision)
        self.scaler: Optional[torch.cuda.amp.GradScaler]
        
        if self.autocast_dtype is torch.float16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.global_step = 0

    def train(self) -> None:
        """Run training loop."""
        self.model.train()

        for epoch in range(self.cfg.training.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.autocast_dtype is not None and self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        loss = self.model(**batch).loss
                else:
                    loss = self.model(**batch).loss

                loss = loss / self.cfg.training.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.cfg.training.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.training.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.training.max_grad_norm
                        )
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    if self.global_step % self.cfg.training.logging_steps == 0:
                        print(
                            f"Epoch {epoch} | step {self.global_step} | "
                            f"loss {loss.item():.4f}",
                        )

                    if self.global_step % self.cfg.training.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    if self.global_step >= self.max_train_steps:
                        return

    def _save_checkpoint(self, step: int) -> None:
        """Save model and tokenizer checkpoint."""
        ckpt_dir = self.output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

