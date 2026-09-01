"""Hydra entrypoint for evaluating a trained VLTA policy.

Loads a checkpoint, runs validation, and optionally samples actions for
visualisation. Run from the repository root::

    python eval.py ckpt_path=checkpoints/step-1000.ckpt
    python eval.py ckpt_path=checkpoints/last.ckpt data.batch_size=32

The evaluation uses the same dataset preprocessing (whitening from
``meta/stats.json``) as training, so metrics are comparable. Action sampling
is performed under ``torch.inference_mode`` explicitly at the call site.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from origami_iros.train.config import TrainConfig, register_config_store

register_config_store()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _eval(cfg)


def _eval(cfg: DictConfig) -> None:
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    from origami_iros.train.datamodule import VLTA_pl_datamodule
    from origami_iros.train.factory import (
        build_preprocessor,
        resolve_data_facts,
        resolve_model_config,
    )
    from origami_iros.train.lightning_module import VLTA_pl_module

    ckpt_path: str | None = cfg.get("ckpt_path", None)  # type: ignore[attr-defined]
    # Remove ckpt_path before converting to typed config (not part of TrainConfig schema)
    if "ckpt_path" in cfg:
        cfg = cfg.copy()
        del cfg["ckpt_path"]
    tcfg: TrainConfig = TrainConfig.from_dictconfig(cfg)

    facts, season_roots = resolve_data_facts(
        tcfg.data.data_root, tcfg.data.dataset_subdir, tcfg.data
    )
    model_config = resolve_model_config(tcfg.model, facts)
    preprocessor = build_preprocessor(season_roots, facts, tcfg.data)

    datamodule = VLTA_pl_datamodule(tcfg.data, fps=facts.fps, preprocessor=preprocessor)

    if ckpt_path is not None:
        module = VLTA_pl_module.load_from_checkpoint(
            ckpt_path, model_config=model_config, optimizer_config=tcfg.optimizer, chunk_size=tcfg.data.chunk_size
        )
    else:
        module = VLTA_pl_module(model_config, tcfg.optimizer, chunk_size=tcfg.data.chunk_size)

    logger = CSVLogger("./logs_eval")
    trainer = L.Trainer(
        accelerator=tcfg.accelerator,
        devices=tcfg.devices,
        precision=tcfg.precision,
        logger=logger,
        num_sanity_val_steps=0,
    )

    # Standard validation loop
    trainer.validate(module, datamodule=datamodule)

    # Additional inference sampling for qualitative checks (explicit inference_mode)
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    module.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            obs, action, _ = batch
            pred = module.model.sample_actions(obs)
            # pred is (batch, chunk, action_dim) whitened; unnormalize for inspection if needed
            print(f"batch {batch_idx}: pred {tuple(pred.shape)} target {tuple(action.shape)}")
            if batch_idx >= 2:
                break


if __name__ == "__main__":
    main()
