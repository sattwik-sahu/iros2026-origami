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
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.loggers import WandbLogger

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

    datamodule = VLTA_pl_datamodule(
        tcfg.data, fps=facts.fps, preprocessor=preprocessor
    )
    module = VLTA_pl_module(
        model_config, tcfg.optimizer, chunk_size=tcfg.data.chunk_size
    )

    wandb_logger = WandbLogger(
        project=tcfg.wandb.project,
        entity=tcfg.wandb.entity,
        name=tcfg.wandb.name or tcfg.run_name,
        tags=list(tcfg.wandb.tags),
        save_code=tcfg.wandb.save_code,
    )
    csv_logger = CSVLogger("./logs")

    callbacks = [
        ActionSampleLogger(every_n_steps=tcfg.callbacks.val_every_n_steps),
        build_checkpoint_callback(
            every_n_steps=tcfg.callbacks.checkpoint_every_n_steps,
            dirpath=tcfg.callbacks.checkpoint_dir,
        ),
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
