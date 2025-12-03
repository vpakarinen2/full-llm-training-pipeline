"""Training utilities."""

from __future__ import annotations

import torch
import math

from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

from ai_pipeline.training.optim import create_optimizer, create_scheduler
from ai_pipeline.training.optim import create_optimizer, create_scheduler
from ai_pipeline.data.dataset import JsonlTextDataset, collate_fn
from ai_pipeline.config.schema import FullConfig, ModelConfig
from ai_pipeline.data.tokenization import create_tokenizer
from ai_pipeline.models.factory import create_causal_lm
from ai_pipeline.utils.logging import get_logger


logger = get_logger(__name__)


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
    """Simple trainer for causal LM."""
    def __init__(self, cfg: FullConfig, resume_from: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.device = _get_device(cfg)

        logger.info("Initializing trainer on device: %s", self.device)

        self.output_dir = cfg.training.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume_from is not None:
            resume_model_cfg = ModelConfig(
                model_name=str(resume_from),
                torch_dtype=cfg.model.torch_dtype,
                gradient_checkpointing=cfg.model.gradient_checkpointing,
                use_flash_attention=cfg.model.use_flash_attention,
                max_position_embeddings=cfg.model.max_position_embeddings,
            )
            self.tokenizer = create_tokenizer(resume_model_cfg)
            self.model = create_causal_lm(resume_model_cfg)
        else:
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

        if len(self.train_dataloader) == 0:
            raise ValueError("Training dataloader is empty; check train_path and dataset contents.")

        dataset_size = len(self.train_dataset)
        logger.info(
            "Loaded training dataset from %s with %d examples",
            self.cfg.data.train_path,
            dataset_size,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / cfg.training.gradient_accumulation_steps
        )
        self._num_update_steps_per_epoch = num_update_steps_per_epoch
        self.max_train_steps = cfg.training.num_epochs * num_update_steps_per_epoch

        logger.info(
            "Training for %d epochs (%d update steps per epoch, %d total update steps)",
            self.cfg.training.num_epochs,
            self._num_update_steps_per_epoch,
            self.max_train_steps,
        )

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
        self._start_epoch = 0

        if resume_from is not None:
            self._load_training_state(resume_from)

    def train(self) -> None:
        """Run training loop."""
        self.model.train()

        for epoch in range(self._start_epoch, self.cfg.training.num_epochs):
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
                        logger.info(
                            "Epoch %d | step %d | loss %.4f",
                            epoch,
                            self.global_step,
                            loss.item(),
                        )

                    if self.global_step % self.cfg.training.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    if self.global_step >= self.max_train_steps:
                        return

    def _load_training_state(self, checkpoint_dir: Path) -> None:
        """Load training state from checkpoint directory."""
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        logger.info("Resuming training from %s", checkpoint_dir)

        state_path = checkpoint_dir / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu")
            self.global_step = int(state.get("global_step", 0))
            if self._num_update_steps_per_epoch > 0:
                self._start_epoch = self.global_step // self._num_update_steps_per_epoch

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.is_file():
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

        scaler_path = checkpoint_dir / "scaler.pt"
        if scaler_path.is_file() and self.scaler is not None:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))

        logger.info(
            "Loaded training state: global_step=%d, start_epoch=%d",
            self.global_step,
            self._start_epoch,
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save model and tokenizer checkpoint."""
        ckpt_dir = self.output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        if self.scaler is not None:
            torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")

        torch.save({"global_step": self.global_step}, ckpt_dir / "training_state.pt")

        logger.info("Saved checkpoint to %s (step %d)", ckpt_dir, step)
