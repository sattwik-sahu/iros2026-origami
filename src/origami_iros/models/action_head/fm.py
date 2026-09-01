"""Flow-matching action head.

This module implements the action head for the VLTA policy using the reference
``flow_matching`` library (from the original Facebook / Lipman et al. authors)
to perform conditional optimal-transport flow matching on action chunks.

The module is responsible for the *flow-matching machinery* only:

* the conditional-OT probability path,
* time sampling from a Beta prior (per the original flow-matching paper),
* the ODE solver used to integrate the learned velocity field back to a clean
  action chunk.

The learned *velocity network* itself is transport-agnostic and lives in
:class:`origami_iros.models.action_head.velocity_transformer.ActionDiT`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from typing import override

from origami_iros._typing import Action
from origami_iros.models.action_head.velocity_transformer import ActionDiT
from origami_iros.models.base import BaseActionModule


class WrappedVelocityModel(ModelWrapper):
    """Adapts :class:`ActionDiT` to the ``flow_matching`` solver interface.

    The ``flow_matching`` solvers expect a model whose ``forward`` accepts
    ``(x, t)`` positional arguments (plus any extra keyword tensors). This
    wrapper routes those calls to the transformer, injecting the observation
    memory through the ``**extras`` channel as ``memory``.
    """

    @override
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, **extras: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity given a noisy action chunk and time.

        Args:
            x: Noisy action chunk of shape ``(batch, chunk_size, action_dim)``.
            t: Flow time for each batch element, shape ``(batch,)``.
            **extras: Must contain the observation memory under the ``memory``
                key, of shape ``(batch, n_tokens, hidden_dim)``.

        Returns:
            Predicted velocity of shape ``(batch, chunk_size, action_dim)``.
        """
        memory = extras["memory"]
        return self.model(x, t, memory)


class FlowMatchingActionHead(BaseActionModule[torch.Tensor]):
    """Conditional-OT flow-matching action head.

    Maps an encoded observation to a chunk of actions by learning a velocity
    field that transports an isotropic Gaussian ``x_0`` to the target action
    chunk ``x_1``, integrated with an ODE solver at inference time.

    The probability path, time scheduler and ODE solver are provided by the
    reference ``flow_matching`` library; the transformer backbone is
    :class:`ActionDiT`.

    Attributes:
        chunk_size: Number of action timesteps in a chunk.
        action_dim: Dimensionality of a single action vector.
        hidden_dim: Transformer embedding dimension.
        num_inference_steps: Number of Euler steps at sampling time.
    """

    def __init__(
        self,
        chunk_size: int = 13,
        action_dim: int = 65,
        dim_in: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_inference_steps: int = 10,
    ) -> None:
        """Initialise the action head.

        Args:
            chunk_size: Number of action timesteps in a chunk.
            action_dim: Dimensionality of a single action vector.
            dim_in: Dimension of the observation tokens feeding the head.
            hidden_dim: Embedding dimension for the transformer tokens.
            num_layers: Number of alternating cross/causal-attention blocks.
            num_heads: Number of attention heads in each block.
            num_inference_steps: Number of Euler integration steps at sampling.
        """
        super().__init__(dim_action=chunk_size * action_dim)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps

        # Flow-matching reference machinery.
        self.prob_path = CondOTProbPath()
        self.scheduler = CondOTScheduler()

        # Transport-agnostic velocity network.
        self.velocity_model = ActionDiT(
            chunk_size=chunk_size,
            action_dim=action_dim,
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # ODE solver with wrapped velocity model.
        self.solver = ODESolver(velocity_model=WrappedVelocityModel(self.velocity_model))

    def predict_velocity(
        self, noisy_action: torch.Tensor, t: torch.Tensor, obs_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity using the wrapped network.

        Args:
            noisy_action: Noisy action chunk of shape
                ``(batch, chunk_size, action_dim)``.
            t: Flow time for each batch element, shape ``(batch,)``.
            obs_tokens: Encoded observation tokens of shape
                ``(batch, n_tokens, dim_in)``.

        Returns:
            Predicted velocity of shape ``(batch, chunk_size, action_dim)``.
        """
        memory = self.velocity_model.memory_from_tokens(obs_tokens)
        return self.velocity_model(noisy_action, t, memory)

    def compute_loss(
        self,
        x: torch.Tensor,
        target_action: torch.Tensor,
        noise: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Conditional-OT flow-matching loss.

        Samples a flow time ``t`` from a Beta prior, builds the interpolated
        sample ``x_t = (1 - t) x_0 + t x_1`` via the reference probability path,
        and regresses the predicted velocity against the conditional target
        velocity ``x_1 - x_0`` with MSE.

        Args:
            x: Encoded observation tokens of shape ``(batch, n_tokens, dim_in)``.
            target_action: Target action chunk of shape ``(batch, chunk*action_dim)``.
            noise: Optional fixed noise ``x_0``; defaults to isotropic Gaussian.
            loss_mask: Optional boolean padding mask of shape ``(batch, chunk)``;
                ``False`` entries are masked out of the loss.

        Returns:
            Scalar MSE loss.
        """
        batch_size = x.size(0)
        device = x.device
        target_action = target_action.reshape(batch_size, self.chunk_size, self.action_dim)

        # Time sampling follows the original flow-matching paper (Beta prior).
        t = self._sample_time(batch_size, device)

        x_0 = torch.randn_like(target_action) if noise is None else noise
        path_sample = self.prob_path.sample(x_0=x_0, x_1=target_action, t=t)

        memory = self.velocity_model.memory_from_tokens(x)
        v_pred = self.velocity_model(path_sample.x_t, t, memory)

        if loss_mask is None:
            return F.mse_loss(v_pred, path_sample.dx_t)

        per_elem = F.mse_loss(v_pred, path_sample.dx_t, reduction="none")
        mask = (~loss_mask).float().unsqueeze(-1)
        return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)

    def sample_actions(
        self,
        obs_tokens: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample an action chunk by integrating the learned velocity field.

        Starts from isotropic Gaussian noise (or a provided ``noise``) and
        integrates the velocity field from ``t=0`` to ``t=1`` with the reference
        ODE solver using Euler's method.

        Args:
            obs_tokens: Encoded observation tokens of shape
                ``(batch, n_tokens, dim_in)``.
            noise: Optional starting sample ``x_0``; defaults to Gaussian noise.
            num_inference_steps: Optional number of Euler steps; overrides the
                value set at construction time.

        Returns:
            Sampled action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        batch_size = obs_tokens.size(0)
        device, dtype = obs_tokens.device, obs_tokens.dtype
        num_steps = num_inference_steps or self.num_inference_steps

        x_init = (
            torch.randn(
                batch_size, self.chunk_size, self.action_dim, device=device, dtype=dtype
            )
            if noise is None
            else noise
        )

        time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        memory = self.velocity_model.memory_from_tokens(obs_tokens)
        sol = self.solver.sample(
            x_init=x_init,
            time_grid=time_grid,
            step_size=1.0 / num_steps,
            method="euler",
            memory=memory,
        )
        return sol

    @override
    def forward(self, x: torch.Tensor) -> Action:
        """Sample an action chunk for the given encoded observation.

        Args:
            x: Encoded observation tokens of shape ``(batch, n_tokens, dim_in)``.

        Returns:
            Predicted action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        return self.sample_actions(x)

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample flow times from a Beta prior.

        Uses the Beta(1.5, 1.0) prior from the original flow-matching paper,
        rescaled to keep ``t`` strictly inside ``(0, 1)`` to avoid boundary
        pathologies.

        Args:
            batch_size: Number of independent time samples.
            device: Torch device for the returned tensor.

        Returns:
            Sampled times of shape ``(batch_size,)`` in ``[0.001, 0.999]``.
        """
        alpha, beta, scale, offset = 1.5, 1.0, 0.999, 0.001
        t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        return t * scale + offset
