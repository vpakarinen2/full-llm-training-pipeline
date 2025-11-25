# Full LLM Training Pipeline

## Workflow

#### data → config → train → evaluate → compare

## Setup

### Create Python Environment

```
python3.11 -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

## Run

### Training

```
python scripts/train.py --config configs/train_example.yaml --device auto
```

### Evaluation

```
python scripts/evaluate.py --config configs/train_example.yaml

python scripts/evaluate.py --config configs/train_example.yaml --checkpoint configs/outputs/example_run/step_2 (checkpoint)
```

## Config

### Key Settings

- `model_name`: base model identifier (e.g. `gpt2`)
- `torch_dtype`: numeric precision (e.g. `float32`, `bf16`)
- `train_path`, `val_path`: paths to JSONL files under `data/raw/`
- `max_seq_length`: maximum sequence length in tokens
- `instruction_field`, `input_field`, `output_field`: names of JSON keys to build prompts
- `output_dir`: where checkpoints and logs are written
- `num_epochs`, `train_batch_size`, `gradient_accumulation_steps`
- `learning_rate`, `lr_scheduler_type`, `warmup_steps` / `warmup_ratio`
- `logging_steps`, `save_steps`, `save_total_limit`
- `max_eval_samples`: limit on eval set size
- `max_new_tokens`, `temperature`, `top_p`, `top_k`, `num_beams`: generation settings controlling the model output
- `save_predictions`, `predictions_filename`: write out generated outputs and JSONL file name to use
- `seed`: random seed for reproducibility
- `device`: which hardware to run on (`cuda`, `cpu`, or `auto`)

## CLI Arguments

- `--config`: path to YAML/JSON config file (e.g. `configs/train_example.yaml`).
- `--seed`: set the random seed used for current run.
- `--device`: select the hardware backend for training (e.g. `cuda`, `cpu`, `auto`).
- `--checkpoint`: path to model checkpoint directory.

## Author

Ville Pakarinen (@vpakarinen2)
