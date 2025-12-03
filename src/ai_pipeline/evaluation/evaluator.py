"""Evaluation utilities."""

from __future__ import annotations

import torch
import json

from typing import Any, Dict, List, Mapping, Optional
from pathlib import Path

from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from ai_pipeline.data.tokenization import create_tokenizer, format_example
from ai_pipeline.config.schema import DataConfig, EvalConfig, FullConfig
from ai_pipeline.evaluation.metrics import compute_perplexity
from ai_pipeline.utils.logging import get_logger


logger = get_logger(__name__)


def _get_device(cfg: FullConfig) -> torch.device:
    if cfg.run.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.run.device)


def _load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Mapping[str, Any]]:
    examples: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


class Evaluator:
    """Evaluator for causal LM."""
    def __init__(
        self,
        cfg: FullConfig,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.device = _get_device(cfg)

        logger.info("Initializing evaluator on device: %s", self.device)

        self.tokenizer: PreTrainedTokenizerBase = create_tokenizer(cfg.model)

        if checkpoint_dir is not None:
            logger.info("Loading model from checkpoint: %s", checkpoint_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch.bfloat16 if cfg.model.torch_dtype.lower() in {"bf16", "bfloat16"} else None,
                device_map=None,
            )
        else:
            logger.info("Loading base model: %s", cfg.model.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                torch_dtype=None,
                device_map=None,
            )

        self.model.to(self.device)
        self.model.eval()

    def _get_eval_path(self, data_cfg: DataConfig) -> Path:
        if data_cfg.val_path is not None:
            return data_cfg.val_path
        return data_cfg.train_path

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation and return metrics dict."""
        data_cfg = self.cfg.data
        eval_cfg = self.cfg.eval

        data_path = self._get_eval_path(data_cfg)
        if not data_path.is_file():
            raise FileNotFoundError(f"Eval file not found: {data_path}")

        logger.info(
            "Evaluating on file: %s (max_eval_samples=%s)",
            data_path,
            str(eval_cfg.max_eval_samples),
        )

        examples = _load_jsonl(data_path, max_samples=eval_cfg.max_eval_samples)

        logger.info("Loaded %d evaluation examples", len(examples))

        total_loss = 0.0
        total_tokens = 0

        for ex in examples:
            text = format_example(ex, data_cfg)
            encoded = self.tokenizer(
                text,
                max_length=data_cfg.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            encoded["labels"] = encoded["input_ids"].clone()

            output = self.model(**encoded)
            loss = float(output.loss.detach().cpu())

            token_count = int(encoded["input_ids"].numel())
            total_loss += loss * token_count
            total_tokens += token_count

        if total_tokens == 0:
            raise ValueError("No tokens in evaluation set")

        avg_loss = total_loss / total_tokens
        metrics: Dict[str, float] = {"eval_loss": avg_loss}
        metrics["perplexity"] = compute_perplexity(metrics, loss_key="eval_loss")

        if self.cfg.eval.save_predictions:
            self._save_predictions(examples)

        return metrics

    @torch.no_grad()
    def _save_predictions(self, examples: List[Mapping[str, Any]]) -> None:
        eval_cfg: EvalConfig = self.cfg.eval
        data_cfg: DataConfig = self.cfg.data

        out_dir = self.cfg.training.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / eval_cfg.predictions_filename

        logger.info("Saving predictions to %s", out_path)

        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                prompt = format_example(ex, data_cfg)
                encoded = self.tokenizer(
                    prompt,
                    max_length=data_cfg.max_seq_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                generated_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=eval_cfg.max_new_tokens,
                    temperature=eval_cfg.temperature,
                    top_p=eval_cfg.top_p,
                    top_k=eval_cfg.top_k,
                    num_beams=eval_cfg.num_beams,
                )

                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                )

                record = {
                    "prompt": prompt,
                    "generated": generated_text,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
