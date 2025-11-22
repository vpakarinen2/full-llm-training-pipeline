"""Configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Config related to base language model."""
    # Name or path of base the model (Hugging Face or local disk)
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # Torch dtype to use when loading the model
    torch_dtype: str = "bfloat16"

    # Enable gradient checkpointing (reduces memory usage)
    gradient_checkpointing: bool = True

    # Use flash-attention (if supported)
    use_flash_attention: bool = False

    # Max context window size
    max_position_embeddings: int = 4096


@dataclass
class DataConfig:
    """Config for dataset and preprocessing."""
    # Path to training data file (e.g. JSONL, CSV, or text)
    train_path: Path = Path("data/train.jsonl")

    # Path to validation data (optional)
    val_path: Optional[Path] = Path("data/raw/val.jsonl")

    # Training data to reserve for validation (set to 0.0 to disable)
    val_split_ratio: float = 0.0

    # Maximum sequence length (in tokens)
    max_seq_length: int = 2048

    # Pack multiple short sequences
    pack_sequences: bool = True

    # Column names or keys (contains relevant text fields)
    text_field: str = "text"
    instruction_field: Optional[str] = None
    input_field: Optional[str] = None
    output_field: Optional[str] = None

    # Prompt template to use (optional)
    chat_template: Optional[str] = None

    # Number of worker processes
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Config for training loop and optimization."""
    # Output directory for checkpoints and logs
    output_dir: Path = Path("outputs/llm_finetune")

    # Global training parameters
    num_epochs: int = 3
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # Optimization hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0

    # Mixed precision
    mixed_precision: str = "bf16"

    # Log training metrics (in steps)
    logging_steps: int = 10

    # Run evaluation (in steps)
    eval_steps: int = 200

    # Save model checkpoints (in steps)
    save_steps: int = 500

    # Maximum number of checkpoints
    save_total_limit: int = 3

    # Load best checkpoint
    load_best_model_at_end: bool = True

    # Metric to monitor when selecting best model
    metric_for_best_model: str = "eval_loss"

    # Higher metric values correspond to better performance
    greater_is_better: bool = False


@dataclass
class EvalConfig:
    """Config specific to evaluation and prediction."""
    # Max number of evaluation examples
    max_eval_samples: Optional[int] = None

    # Parameters for text generation-based eval
    max_new_tokens: int = 256
    temperature: float = 0.7
    num_beams: int = 1
    top_p: float = 0.9
    top_k: int = 50

    # Save generated predictions
    save_predictions: bool = True

    # Path where predictions will be stored
    predictions_filename: str = "predictions.jsonl"


@dataclass
class RunConfig:
    """Global run configuration."""
    # Random seed
    seed: int = 42

    # Device to use for training and eval
    device: str = "auto"

    # Enable deterministic algorithms
    deterministic: bool = False

    # Set experiment name (optional)
    experiment_name: Optional[str] = None

    # Tag list for experiment tracking
    tags: List[str] = field(default_factory=list)


@dataclass
class FullConfig:
    """Top-level configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
