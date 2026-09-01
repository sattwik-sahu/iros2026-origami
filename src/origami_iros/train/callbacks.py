"""Lightning callbacks for action sampling and checkpointing."""

from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from typing import override


class ActionSampleLogger(Callback):
    """Log sampled action chunks to WandB every ``every_n_steps`` optimizer steps.

    Replaces the hand-rolled sampling evaluation in the old training loop. On
    every Nth optimizer step it draws a single batch from the validation loader,
    samples a chunk of actions with the current policy, and logs the sampled
    trajectory plus the ground-truth target as arrays.

    Attributes:
        every_n_steps: Log sampled actions every this many optimizer steps.
        max_batches: Maximum number of validation batches sampled at once.
    """

    def __init__(self, every_n_steps: int, max_batches: int = 2) -> None:
        """Initialise the callback.

        Args:
            every_n_steps: Log sampled actions every this many optimizer steps.
            max_batches: Maximum number of validation batches sampled at once.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches

    @override
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Sample actions for the current validation batch.

        Args:
            trainer: The Lightning trainer.
            pl_module: The VLTA Lightning module.
            outputs: Output of ``validation_step`` (unused).
            batch: The validation batch ``(obs, action, action_is_pad)``.
            batch_idx: Index of the batch within the epoch.
        """
        if batch_idx >= self.max_batches:
            return
        global_step = trainer.global_step
        if global_step % self.every_n_steps != 0:
            return

        obs, action, _ = batch
        pl_module.eval()
        with torch.no_grad():
            pred = pl_module.sample_for_logging(obs)
        pl_module.train()

        action = action.reshape(pred.shape)
        pred = pred.cpu().numpy()
        action = action.cpu().numpy()

        logger = trainer.logger
        if logger is not None and hasattr(logger, "experiment"):
            experiment = logger.experiment
            for i in range(min(pred.shape[0], 4)):
                experiment.log(
                    {
                        f"replay/batch{batch_idx}_sample{i}/pred": pred[i],
                        f"replay/batch{batch_idx}_sample{i}/target": action[i],
                        "step": global_step,
                    }
                )
        elif trainer.logger is not None:
            trainer.logger.log_metrics(
                {
                    f"replay/batch{batch_idx}_sample_mean": float(pred.mean()),
                },
                step=global_step,
            )


def build_checkpoint_callback(every_n_steps: int, dirpath: str) -> ModelCheckpoint:
    """Create a ``ModelCheckpoint`` saving every ``every_n_steps`` steps.

    This replaces the manual ``torch.save(...)`` calls in the old training loop.

    Args:
        every_n_steps: Save a checkpoint every this many optimizer steps.
        dirpath: Directory in which checkpoints are stored.

    Returns:
        A configured :class:`ModelCheckpoint` callback.
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        filename="step-{step}",
        every_n_train_steps=every_n_steps,
        save_top_k=-1,
    )
