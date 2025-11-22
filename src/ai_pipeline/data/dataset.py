"""Dataset utilities for full LLM fine-tuning."""

from __future__ import annotations

import torch
import json

from typing import Any, Dict, List, Mapping, Sequence
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path

from ai_pipeline.data.tokenization import format_example
from ai_pipeline.config.schema import DataConfig


@dataclass
class JsonlDatasetConfig:
    """Config for JSONL dataset source."""
    path: Path


class JsonlTextDataset(Dataset):
    """Simple JSONL dataset using `DataConfig`."""
    def __init__(self, data_cfg: DataConfig) -> None:
        self.data_cfg = data_cfg
        self.path = data_cfg.train_path

        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self._examples: List[Mapping[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self._examples[idx]


def collate_fn(
    batch: Sequence[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: DataConfig,
) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    texts: List[str] = [format_example(ex, data_cfg) for ex in batch]

    tokenized = tokenizer(
        texts,
        max_length=data_cfg.max_seq_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized
