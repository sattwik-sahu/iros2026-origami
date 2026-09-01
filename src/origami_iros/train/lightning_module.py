"""PyTorch Lightning module wrapping the VLTA policy for training.

The module is deliberately thin: it builds the policy through the Hydra-composed
``_target_`` factory (:func:`~origami_iros.models.factory.build_vlta_policy`),
defines the train/validation steps, and configures the optimizer. It subclasses
:class:`lightning.LightningModule` and uses the ``@override`` decorator on every
re-implemented hook.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from typing import override
from transformers import get_cosine_schedule_with_warmup

from origami_iros.models._typing import Observation
from origami_iros.models.factory import build_vlta_policy
from origami_iros.train.config import ModelConfig, OptimizerConfig


class LRScheduleOptimizer(torch.optim.Optimizer):
    """A ``torch.optim.Optimizer`` that also steps a learning-rate scheduler.

    Lightning's :meth:`configure_optimizers` may return a single
    ``torch.optim.Optimizer``; returning a dict is discouraged. This wrapper owns
    both the base optimizer and its LR scheduler, and advances the scheduler
    whenever :meth:`step` is called, so the cosine-with-warmup schedule is applied
    while still returning a plain torch optimizer. Uses delegation to the inner
    optimizer for all standard optimizer attributes.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: Any) -> None:
        """Initialise the wrapper.

        Args:
            optimizer: The underlying torch optimizer.
            scheduler: Any object exposing a ``step()`` method.
        """
        self._inner = optimizer
        self._scheduler = scheduler
        # Initialise base Optimizer with the inner's param groups so that
        # ``isinstance(..., Optimizer)`` checks and Lightning's handling work.
        super().__init__(optimizer.param_groups, {})

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to the inner optimizer.

        Args:
            name: Attribute name.

        Returns:
            The attribute on the inner optimizer.
        """
        if name in {"_inner", "_scheduler"}:
            raise AttributeError(name)
        return getattr(self._inner, name)

    @property
    def param_groups(self):  # type: ignore[override]
        """Proxy to the inner optimizer's parameter groups."""
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._inner.param_groups = value

    @override
    def step(self, closure=None):  # type: ignore[override]
        """Step the inner optimizer, then the LR scheduler.

        Args:
            closure: Optional closure passed to the optimizer step.

        Returns:
            The result of the inner optimizer step.
        """
        out = self._inner.step(closure)
        self._scheduler.step()
        return out

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        """Proxy :meth:`zero_grad` to the inner optimizer.

        Args:
            set_to_none: Passed through to the inner optimizer.
        """
        self._inner.zero_grad(set_to_none=set_to_none)


class VLTA_pl_module(L.LightningModule):
    """Lightning module for the VLTA flow-matching policy.

    Attributes:
        model: The underlying :class:`VLTAPolicy` network.
        chunk_size: Number of action timesteps predicted per chunk.
        model_config: Model hyperparameters.
        optimizer_config: Optimizer and LR-schedule hyperparameters.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        chunk_size: int,
        action_normalizer: Any | None = None,
    ) -> None:
        """Initialise the module.

        Args:
            model_config: Model architecture hyperparameters (already resolved
                with data facts from the dataset metadata).
            optimizer_config: Optimizer and learning-rate schedule hyperparameters.
            chunk_size: Number of action timesteps predicted per chunk. Must
                match the window used to build the dataset's ``delta_timestamps``.
            action_normalizer: Optional normalizer for feasible clamping (from
                stats.json). If provided, it is passed to the policy so that
                ``sample_feasible_actions`` is available and logging can report
                feasible vs raw.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.chunk_size = chunk_size
        self.action_normalizer = action_normalizer

        self.model = build_vlta_policy(model_config, chunk_size, action_normalizer=action_normalizer)

    @override
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: A ``(obs, action, action_is_pad)`` tuple from the data loader.
            batch_idx: Index of the batch within the epoch.

        Returns:
            The scalar training loss.
        """
        obs, action, action_is_pad = batch
        loss = self.model.compute_loss(obs, action, action_is_pad)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    @override
    def validation_step(self, batch, batch_idx: int) -> None:
        """Run one validation step.

        Args:
            batch: A ``(obs, action, action_is_pad)`` tuple from the data loader.
            batch_idx: Index of the batch within the epoch.
        """
        obs, action, action_is_pad = batch
        loss = self.model.compute_loss(obs, action, action_is_pad)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct the optimizer wrapped with a learning-rate schedule.

        Returns a single :class:`torch.optim.Optimizer` (via
        :class:`LRScheduleOptimizer`) that also advances a cosine-annealing-warmup
        schedule on every ``step()``.

        Returns:
            A torch optimizer that steps the LR schedule alongside the parameters.
        """
        cfg = self.optimizer_config
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.total_steps,
            num_cycles=0.5,
        )
        return LRScheduleOptimizer(optimizer, scheduler)

    @override
    def on_before_optimizer_step(self, optimizer) -> None:
        """Log gradient norm before optimizer step (wandb numeric)."""
        # Compute global grad norm
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if grads:
            grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2)
            self.log("grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)
            # param norm
            param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in self.model.parameters()]), 2)
            self.log("param_norm", param_norm, on_step=True, on_epoch=False)

    @override
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log the current learning rate and additional diagnostics.

        Args:
            outputs: Output of ``training_step``.
            batch: The batch processed by the step.
            batch_idx: Index of the batch within the epoch.
        """
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False)
        # Log loss histograms and action stats every 500 steps
        if batch_idx % 500 == 0:
            obs, action, _ = batch
            # action stats (whitened)
            self.log("action/mean", action.mean(), on_step=True, on_epoch=False)
            self.log("action/std", action.std(), on_step=True, on_epoch=False)
            self.log("action/max", action.max(), on_step=True, on_epoch=False)
            self.log("action/min", action.min(), on_step=True, on_epoch=False)
            # log histograms via wandb (if available)
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "histograms/action_whitened": wandb.Histogram(action.detach().cpu().numpy()),
                        },
                        step=self.global_step,
                    )
            except Exception:
                pass

    @torch.inference_mode()
    def sample_for_logging(self, obs: Observation) -> torch.Tensor:
        """Sample an action chunk for logging/visualisation.

        Runs under ``torch.inference_mode`` since sampled actions are only logged.

        Args:
            obs: A batch of observations.

        Returns:
            Predicted action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        return self.model.sample_actions(obs)
