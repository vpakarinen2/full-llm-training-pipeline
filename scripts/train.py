"""Training entrypoint for full LLM fine-tuning."""

from __future__ import annotations

import os
import argparse
import sys

from pprint import pprint
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def _add_src_to_path() -> None:
    """Ensure `src/` is on sys.path."""
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    if src_dir.is_dir():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

_add_src_to_path()

from ai_pipeline.config.loader import load_config  # noqa: E402
from ai_pipeline.training.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Full LLM fine-tune training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override run.seed (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override run.device (e.g. 'cuda', 'cpu', 'auto')",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.device is not None:
        cfg.run.device = args.device

    print("Loaded config from:", config_path)
    pprint(cfg)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
