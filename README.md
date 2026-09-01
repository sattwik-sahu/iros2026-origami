# Origami IROS — Bimanual VLTA Flow-Matching Policy

Bimanual imitation-learning policy that maps multimodal observations (four camera
views, raw tactile images, proprioceptive state) to chunks of future joint
actions using a conditional-OT flow-matching head.

<a href="https://wandb.ai/building-text/iros2026-origami" target="_blank">
  <img src="https://img.shields.io/badge/WandB-ros2026--origami-brightgreen" alt="WandB">
</a>

## Architecture

```mermaid
flowchart LR
    subgraph Dataset["Dataset (LeRobot v3.0)"]
        A[Seasons<br/>dataset/lerobot3.0/<br/>meta + videos + data]
        B[info.json / stats.json<br/>auto-derived facts]
        A --> B
    end

    subgraph Preproc["Preprocessing"]
        C[Whitening<br/>mean/std from stats.json]
        D[Image Normalize<br/>raw tactile 480x1600]
        B --> C
        B --> D
    end

    subgraph Encoders["Observation Encoders (Hydra _target_)"]
        E[Camera<br/>DINOv2 ViT<br/>4 views → tokens]
        F[Tactile<br/>TinyViT<br/>raw primary (480×1600)]
        G[Proprio<br/>MLP ×3 → tokens]
        C --> G
        D --> F
    end

    subgraph Policy["VLTAPolicy (Hydra _target_)"]
        H[VLTA_Encoder<br/>camera + tactile + state]
        I[FlowMatchingActionHead<br/>CondOT + ODESolver + ActionDiT]
        H --> I
        E --> H
        F --> H
        G --> H
    end

    subgraph Train["Training (Lightning + Hydra + WandB)"]
        J[VLTA_pl_module<br/>training_step + logging]
        K[VLTA_pl_datamodule<br/>auto fps/dims + normalizer]
        J --> L[WandB + Checkpoints]
        K --> J
        I --> J
    end

    subgraph Hub["Hugging Face Hub"]
        M[model.safetensors<br/>config.json<br/>normalizer.json<br/>README.md]
        L --> M
    end
```

### Key Features

- **Data facts auto-derived**: fps, action_dim, image sizes, tactile_dim are read from
  `meta/info.json` at runtime — never hard-coded in Hydra config.
- **Whitening**: Actions/state are whitened using pooled stats from `meta/stats.json`
  (`min_std=0.05` clip, `q01/q99` quantiles for feasible clamping).
- **Encoders + action head**: Injected into `VLTAPolicy` as Hydra `_target_` objects
  via `src/origami_iros/models/factory.py`.
- **Training**: Lightning + Hydra + WandB; `configure_optimizers` returns a
  `torch.optim.AdamW` with cosine-annealing-warmup scheduler (`lr=1e-4`,
  `warmup_steps=500`, `lr_min=3e-6`).
- **Loss type**: Configurable via `model.loss_type` (`"mse"` default or `"l1"`).
- **Gradient accumulation**: `accumulate_grad_batches=2` doubles effective batch
  size without increasing VRAM per step.
- **Hub push**: Toggled via `hub.push=true`; uploads
  `model.safetensors` + `config.json` + `normalizer.json` to the repo.
- **WandB logging every step**: `train/loss`, `lr`, `grad_norm`, `param_norm`,
  action histograms (`mean`, `std`, `max`, `min`), and sampled/replay plots.

## Installation

```bash
uv sync
# or
pip install -e .
```

Requirements: Python ≥3.12, PyTorch, `flow-matching`, `lerobot`, `lightning`,
`hydra-core`, `wandb`, `transformers`, `huggingface_hub`, `safetensors`.

## Dataset Layout

```
dataset/
  season_POC22032_.../lerobot3.0/
    meta/info.json      # fps, features, shapes
    meta/stats.json     # mean/std for whitening
    meta/modality.json
    videos/             # head_left/right, wrist_left/right, tactile_raw
    data/chunk-000/
```

All seasons are discovered automatically; `fps` and dimensions are derived from
the first season's metadata.

> **Note**: `tactile_deform` is all zeros in the current recordings. The policy
> uses `tactile_raw` (480×1600) as the primary tactile stream.

## Training

Run training from the repository root:

```bash
# Default: all seasons, auto-derived dims, bs=24, workers=8
python train.py

# Override hyperparameters (example: bs=32, accumulate 2 steps → eff bs=64)
python train.py data.batch_size=32 optimize.accumulate_grad_batches=2

# Debug CPU only
python train.py debug=cpu

# Push to Hugging Face Hub at end
huggingface-cli login
python train.py hub.push=true hub.repo_id=your-org/vlta-north-ces hub.private=false
```

### Configurable Hyperparameters (Hydra)

| Group | Parameter | Default | Description |
|---|---|---|---|
| `data` | `batch_size` | 24 | Per‑GPU batch size (~3 GB VRAM on A4500) |
|  | `num_workers` | 8 | DataLoader workers |
|  | `pin_memory` | True | Pin memory for faster GPU transfer |
|  | `prefetch_factor` | 2 | Prefetch batches per worker |
|  | `normalize_actions` | True | Whitten actions using stats.json |
| `model` | `loss_type` | "mse" | `"mse"` or `"l1"` (see below) |
|  | `hidden_dim` | 512 | Transformer embedding dim |
|  | `action_num_layers` | 6 | Number of transformer layers |
| `optimizer` | `lr` | 1e-4 | AdamW learning rate |
|  | `warmup_steps` | 500 | Linear warmup steps (2.5% of 20k) |
|  | `warmup_start_lr` | 3e-6 | Start LR for warmup |
|  | `lr_min` | 3e-6 | Minimum LR at end of cosine cycle |
|  | `weight_decay` | 1e-4 | L2 penalty |
|  | `grad_clip` | 1.0 | Gradient clipping norm |
| `train` | `max_steps` | 20000 | Total optimizer steps |
|  | `accumulate_grad_batches` | 2 | Effective batch = bs × accumulate |
|  | `num_nodes` | 1 | Number of nodes (for DDP) |
| `wandb` | `project` | "iros2026-origami" | WandB project name |
|  | `entity` | "building-text" | WandB entity |
| `hub` | `push` | True | Push to HF Hub at end |
|  | `repo_id` | "sattwik21/sharpa-north-origami" | HF repo id |

### Loss Type: MSE vs L1

Set `model.loss_type: l1` in the Hydra config (or via command line) to use
L1 loss instead of the default MSE. L1 tends to produce sparser errors and can
help when the model needs to match the exact magnitude of actions rather than
minimizing squared error.

```bash
python train.py model.loss_type=l1
```

### WandB Logging (per training step)

The following metrics are logged at every optimizer step:

- `train/loss` — flow-matching loss (MSE or L1)
- `lr` — current learning rate
- `grad_norm` — global gradient norm
- `param_norm` — global parameter norm
- `action/mean` / `action/std` / `action/max` / `action/min` — whitened action stats
- `histograms/action_whitened` — per-step histogram of whitened actions (logged
  every 500 steps via WandB `Histogram`)

Additionally, the `ActionSampleLogger` callback samples a validation batch every
`val_every_n_steps` and logs:

- `replay/batch{N}_sample{i}/pred` — predicted action chunk (whitened)
- `replay/batch{N}_sample{i}/target` — ground-truth action chunk (whitened)

This lets you visually inspect how close the model's predictions are to the
actual actions at any point during training.

## Evaluation

```bash
# Validate with a checkpoint
python eval.py ckpt_path=checkpoints/step-5000.ckpt

# Sample a few batches for qualitative checks
python eval.py ckpt_path=checkpoints/last.ckpt data.batch_size=8
```

Checkpoints are saved every `callbacks.checkpoint_every_n_steps` (default 1000)
as `step-{step}.ckpt`. The evaluation script loads the checkpoint, runs
`trainer.validate`, and samples actions under `torch.inference_mode` for qualitative
checks.

## Using a Pretrained Model (Hub)

After `hub.push=true`, the repo contains:

- `model.safetensors` — policy state dict (trained weights)
- `config.json` — model architecture + resolved data facts
- `normalizer.json` — action mean/std/q01/q99 for un-whitening
- `README.md` — hub usage instructions

### Loading and Running Inference

```python
import json, torch
from pathlib import Path
from safetensors.torch import load_file
from origami_iros.models.factory import build_vlta_policy
from origami_iros.train.config import ModelConfig

# 1. Reconstruct policy from saved config
cfg = json.loads(Path("config.json").read_text())
model_cfg = ModelConfig(**cfg["model"])
policy = build_vlta_policy(model_cfg, chunk_size=cfg["data_facts"]["chunk_size"])
policy.load_state_dict(load_file(Path("model.safetensors")))

policy.eval()

# 2. Get a batch of observations (use the dataset or your own obs dict)
#    observation format matches what the dataset produces:
#    {"observation.state": ..., "action": ..., "observation.images.tactile_raw": ...}

# 3. Sample actions (whitened)
with torch.inference_mode():
    pred_whitened = policy(obs)  # (B, chunk, 65)

# 4. Un-normalize to robot units using the saved normalizer
norm = json.loads(Path("normalizer.json").read_text())
mean = torch.tensor(norm["action"]["mean"])
std = torch.tensor(norm["action"]["std"])
pred_robot = pred_whitened * std + mean  # (B, chunk, 65) in robot units
```

> **Important**: The policy is trained on whitened actions. Always un-normalize
> with the statistics from `normalizer.json` before sending actions to the robot.
> The `normalizer.json` contains `action["mean"]`, `action["std"]`, and optional
> `q01`/`q99` quantiles for feasible clamping.

### Feasible Action Clamping (at inference)

To clamp predicted actions to joint limits, use the `sample_feasible_actions`
method:

```python
feasible = policy.sample_feasible_actions(obs)  # returns robot units
```

This internally un-whitens, clamps to `q01`/`q99` from `stats.json`, and re-
normalizes so the returned tensor stays in whitened space but is guaranteed
feasible after un-normalization.

## Project Structure

```
configs/                   # Hydra configs (data/model/optimizer/callbacks/wandb/hub)
src/origami_iros/
  data/                    # metadata, preprocessing, dataset, collate
  models/
    encoders/              # camera, tactile, proprio
    action_head/           # flow-matching + velocity transformer
    policy/                # VLTAPolicy (takes encoder + action_head via _target_)
    factory.py             # Hydra _target_ builders
  train/                   # LightningModule, DataModule, callbacks, hub push
train.py                   # Hydra training entrypoint (root)
eval.py                    # Hydra evaluation entrypoint (root)
pyproject.toml             # project deps + console scripts (origami-train, origami-eval)
```

## Implementation Notes

- `VLTAPolicy.forward` delegates to `sample_actions` without an internal
  `torch.no_grad()` — callers wrap inference in `torch.inference_mode()` /
  `torch.no_grad()` explicitly (see `LightningModule.sample_for_logging`).
- `LightningModule.configure_optimizers` returns a `torch.optim.Optimizer`
  (`LRScheduleOptimizer` wrapping AdamW + cosine-with-warmup).
- All overridden base / Lightning hooks use `from typing import override`
  (built-in, Python 3.12).
- Data facts (fps, dims) are resolved once at `setup` time and cached; the
  datamodule uses `pin_memory=True`, `prefetch_factor=2`,
  `persistent_workers=True` for high-throughput loading.
- The `ActionSampleLogger` callback logs sampled predictions and ground-truth
  to WandB on every `every_n_steps` (default 1000/`val_every_n_steps`), giving
  a real-time view of prediction quality throughout training.

## Multi-GPU (Single Node)

For single-A4500 training the default config works out‑of-the-box:

```bash
python train.py accelerator=gpu devices=1
```

Effective batch size is `batch_size × accumulate_grad_batches`. For example,
`batch_size=24` + `accumulate_grad_batches=2` → effective batch 48, VRAM ~5.9 GB,
leaving headroom for optimizer states and activations.

To use gradient accumulation without changing the apparent batch size, keep
`accumulate_grad_batches=1` and set `batch_size` to your target effective size
(e.g., `batch_size=48` for effective 48, still ~5.9 GB VRAM).