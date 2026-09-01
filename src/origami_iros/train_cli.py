"""CLI entrypoint for `origami-train` (installed via pyproject.toml).

This is a thin wrapper around the Hydra training pipeline. It is installed as
``origami-train`` so that ``origami-train data.batch_size=4`` works after
``pip install -e .`` or ``uv sync``. The actual training logic lives in the
root ``train.py`` Hydra entrypoint; this module provides an importable
``main`` for the console script.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from origami_iros.train.config import TrainConfig, register_config_store

register_config_store()

# Hydra config_path must be importable when installed. Try to locate configs
# at repo root (for `origami-train` run from repo root) or via cwd.
from pathlib import Path as _Path

def _find_config_path() -> str:
    # When running from src (dev), configs is at repo root
    candidates = [
        _Path(__file__).resolve().parents[2] / "configs",  # src/origami_iros/train_cli.py -> repo root
        _Path.cwd() / "configs",  # fallback: cwd is repo root
        _Path(__file__).resolve().parents[3] / "configs",  # installed: .venv/.../origami_iros -> repo root may not exist, but try
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "configs"

_CONFIG_PATH = _find_config_path()

HUB_README = """\
# VLTA Flow-Matching Policy (North CES)

Bimanual imitation-learning policy that maps multimodal observations (four camera
views, raw tactile images, and proprioceptive states) to a chunk of future joint
actions using a conditional-OT flow-matching action head.

## Training data

Trained on `north_ces` robot data across multiple collection seasons. Actions are
whitened before training using the dataset statistics shipped in `normalizer.json`.

## Usage

Load the policy, then sample actions for a new observation. The predicted action
chunk is whitened; unnormalize it with the statistics in `normalizer.json` before
sending it to the robot:

```python
import torch
from origami_iros.models.factory import build_vlta_policy
from origami_iros.train.config import ModelConfig
import json
from safetensors.torch import load_file

cfg = json.loads(open("config.json").read())
model_cfg = ModelConfig(**cfg["model"])
policy = build_vlta_policy(model_cfg, chunk_size=cfg["data_facts"]["chunk_size"])
policy.load_state_dict(load_file("model.safetensors"))
policy.eval()
with torch.inference_mode():
    pred_whitened = policy(obs)
norm = json.loads(open("normalizer.json").read())
mean = torch.tensor(norm["action"]["mean"])
std = torch.tensor(norm["action"]["std"])
pred = pred_whitened * std + mean
```

See the project `README.md` for full training/evaluation instructions.
"""


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    _train(cfg)


def _train(cfg: DictConfig) -> None:
    import time

    import lightning as L
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelSummary
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.loggers import WandbLogger

    from origami_iros.train.benchmark import measure_gflops, measure_latency
    from origami_iros.train.callbacks import ActionSampleLogger, build_checkpoint_callback
    from origami_iros.train.datamodule import VLTA_pl_datamodule
    from origami_iros.train.factory import (
        build_preprocessor,
        resolve_data_facts,
        resolve_model_config,
    )
    from origami_iros.train.hub import HubPushCallback
    from origami_iros.train.lightning_module import VLTA_pl_module

    tcfg: TrainConfig = TrainConfig.from_dictconfig(cfg)

    facts, season_roots = resolve_data_facts(
        tcfg.data.data_root, tcfg.data.dataset_subdir, tcfg.data
    )
    model_config = resolve_model_config(tcfg.model, facts)
    preprocessor = build_preprocessor(season_roots, facts, tcfg.data)

    # Build action normalizer for feasible clamping (from pooled stats)
    # Use the same preprocessor's action normalizer if available
    action_normalizer = None
    try:
        # preprocessor stores normalizers dict, get "action"
        action_normalizer = preprocessor._normalizers.get("action")  # type: ignore[attr-defined]
    except Exception:
        action_normalizer = None

    datamodule = VLTA_pl_datamodule(
        tcfg.data, fps=facts.fps, preprocessor=preprocessor
    )
    module = VLTA_pl_module(
        model_config,
        tcfg.optimizer,
        chunk_size=tcfg.data.chunk_size,
        action_normalizer=action_normalizer,
    )

    # --- Latency & GFLOPs benchmark (before training, logged to wandb as numeric) ---
    # Build a dummy batch for benchmark (on GPU if available)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.model.to(device)
        module.model.eval()
        # Create dummy obs with correct shapes (from facts)
        from origami_iros.models._typing import (
            ImageObservation,
            LeftRightImageObservation,
            Observation,
            RobotStateObservation,
            TactileImageObservation,
        )

        B = min(4, tcfg.data.batch_size)
        dummy_obs = Observation(
            image=ImageObservation(
                head=LeftRightImageObservation(
                    left=torch.randn(B, 3, 480, 480, device=device),
                    right=torch.randn(B, 3, 480, 480, device=device),
                ),
                wrist=LeftRightImageObservation(
                    left=torch.randn(B, 3, 480, 480, device=device),
                    right=torch.randn(B, 3, 480, 480, device=device),
                ),
                tactile=TactileImageObservation(
                    deform=torch.randn(B, 3, 480, 1200, device=device),
                    raw=torch.randn(B, 3, 480, 1600, device=device),
                ),
                batch_size=[B],
            ),
            state=RobotStateObservation(
                joint_state=torch.randn(B, facts.joint_state_dim, device=device),
                joint_torque=torch.randn(B, facts.torque_dim, device=device),
                tactile=torch.randn(B, facts.proprio_tactile_dim, device=device),
                batch_size=[B],
            ),
            batch_size=[B],
        )
        lat = measure_latency(module.model, dummy_obs, n_warmup=10, n_iter=30)
        gflops = measure_gflops(module.model, dummy_obs)
        # Print for console and will be logged to wandb after init
        print(f"[benchmark] latency {lat} gflops {gflops}")
        # Store for wandb logging after logger init
        _benchmark_metrics = {**lat, **gflops}
        module.model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"[benchmark] failed: {e}")
        _benchmark_metrics = {}

    wandb_logger = WandbLogger(
        project=tcfg.wandb.project,
        entity=tcfg.wandb.entity,
        name=tcfg.wandb.name or tcfg.run_name,
        tags=list(tcfg.wandb.tags),
        save_code=tcfg.wandb.save_code,
        log_model=False,
    )
    csv_logger = CSVLogger("./logs")

    # Log benchmark as wandb config / summary (numeric)
    if "_benchmark_metrics" in locals() and _benchmark_metrics:
        # wandb can log numeric latency/gflops as config and as first step
        wandb_logger.experiment.config.update(_benchmark_metrics, allow_val_change=True)
        # also log as metrics at step 0
        import wandb as _wandb

        _wandb.log(_benchmark_metrics, step=0)

    callbacks = [
        ActionSampleLogger(every_n_steps=tcfg.callbacks.val_every_n_steps),
        build_checkpoint_callback(
            every_n_steps=tcfg.callbacks.checkpoint_every_n_steps,
            dirpath=tcfg.callbacks.checkpoint_dir,
        ),
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        ModelSummary(max_depth=2),
    ]
    if tcfg.hub.push:
        callbacks.append(
            HubPushCallback(
                repo_id=tcfg.hub.repo_id,
                private=tcfg.hub.private,
                season_root=season_roots[0],
                facts=facts,
                readme=HUB_README,
            )
        )

    trainer = L.Trainer(
        accelerator=tcfg.accelerator,
        devices=tcfg.devices,
        precision=tcfg.precision,
        max_steps=tcfg.max_steps,
        logger=[wandb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=tcfg.callbacks.log_every_n_steps,
        val_check_interval=tcfg.callbacks.val_every_n_steps,
        gradient_clip_val=tcfg.optimizer.grad_clip,
        deterministic=tcfg.deterministic,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
