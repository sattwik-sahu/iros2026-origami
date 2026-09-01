"""Hydra entrypoint for training the VLTA flow-matching policy.

Run from the repository root::

    python train.py

The configuration lives in ``configs/`` and is composed by Hydra. Override any
value from the command line, e.g.::

    python train.py data.batch_size=64 hub.push=true hub.repo_id=me/vlta

At the end of training the model is optionally pushed to the Hugging Face Hub
(see ``configs/hub/`` for the toggle) or used for evaluation.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from origami_iros.train.config import TrainConfig, register_config_store

register_config_store()

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
from origami_iros.models.policy.vlta_policy import VLTAPolicy

model = VLTAPolicy(...)          # build from config.json
model.load_state_dict(torch.load("model.pt")["state_dict"])
model.eval()

with torch.no_grad():
    pred = model.sample_actions(obs)   # (batch, chunk, action_dim), whitened

normalizer = ...                       # from normalizer.json
action = pred * normalizer["action"]["std"] + normalizer["action"]["mean"]
```

See the project `README.md` for full training/evaluation instructions.
"""


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the pipeline from the Hydra config object.

    Args:
        cfg: The composed Hydra configuration.
    """
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

    # Build a typed, validated config from the composed Hydra DictConfig.
    tcfg: TrainConfig = TrainConfig.from_dictconfig(cfg)

    # Derive data facts (fps, dims, image sizes) from the dataset metadata and
    # resolve the preprocessor + model config with those facts.
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
        deterministic=tcfg.seed is not None,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
