"""The full VLTA (vision-language-tactile-action) policy network.

Assembles externally-provided observation encoders and the flow-matching action
head into a single end-to-end :class:`torch.nn.Module`. The encoder and action
head are injected as already-instantiated modules, which lets them be composed
declaratively through Hydra (``_target_``). Exposition is end-to-end: given a
multimodal :class:`Observation`, the policy produces a chunk of future actions.
"""

from __future__ import annotations

import torch
from typing import override
from torch import nn

from typing import Any

from origami_iros.models._typing import Observation, ObservationEncoding
from origami_iros.models.encoders.main import VLTA_Encoder
from origami_iros.models.action_head.fm import FlowMatchingActionHead


class VLTAPolicy(nn.Module):
    """Vision-language-tactile-action imitation-learning policy.

    The policy encodes a multimodal observation (four camera views, tactile
    images, and proprioceptive state) into a token sequence, then uses a
    conditional-OT flow-matching action head to sample a chunk of future
    actions.

    Args:
        encoder: The unified observation encoder (:class:`VLTA_Encoder` or any
            object returning an :class:`ObservationEncoding`).
        action_head: The flow-matching action head.
        vit_dim: Feature dimension of the camera ViT patch tokens.
        tactile_dim: Feature dimension of the tactile tokens.
        hidden_dim: Shared token embedding dimension (the action head's input).
    """

    def __init__(
        self,
        encoder: VLTA_Encoder,
        action_head: FlowMatchingActionHead,
        vit_dim: int,
        tactile_dim: int,
        hidden_dim: int,
        action_normalizer: Any | None = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.action_head = action_head
        self.action_normalizer = action_normalizer

        self.vit_out_proj = nn.Linear(vit_dim, hidden_dim)
        self.tactile_out_proj = nn.Linear(tactile_dim, hidden_dim)

    def encode_observation(self, obs: Observation) -> torch.Tensor:
        """Encode an observation into a sequence of hidden tokens.

        Args:
            obs: The multimodal observation.

        Returns:
            Concatenated hidden tokens of shape ``(batch, n_tokens, hidden_dim)``.
        """
        encoding: ObservationEncoding = self.encoder(obs)
        image_tokens = self.vit_out_proj(encoding.camera_image)
        tactile_tokens = self.tactile_out_proj(encoding.tactile_image)
        proprio_tokens = encoding.state
        return torch.cat([image_tokens, tactile_tokens, proprio_tokens], dim=1)

    def compute_loss(
        self,
        obs: Observation,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the flow-matching training loss.

        Args:
            obs: The multimodal observation.
            target_action: Target action chunk of shape
                ``(batch, chunk_size * action_dim)``.
            action_is_pad: Optional padding mask of shape ``(batch, chunk_size)``.

        Returns:
            Scalar MSE flow-matching loss.
        """
        obs_tokens = self.encode_observation(obs)
        return self.action_head.compute_loss(
            obs_tokens, target_action, loss_mask=action_is_pad
        )

    def sample_actions(
        self, obs: Observation, clamp_feasible: bool = False
    ) -> torch.Tensor:
        """Sample a chunk of actions for an observation.

        This is the differentiable prediction path: gradients flow back through
        the observation encoding and the velocity network. Inference callers
        should wrap the call in ``torch.inference_mode()`` / ``torch.no_grad()``
        explicitly (see :meth:`sample_for_logging` in the Lightning module).

        When ``clamp_feasible`` is True and an ``action_normalizer`` was
        provided at construction, the sampled (whitened) actions are unnormalized
        to robot units, clamped to the feasible ``q01/q99`` limits from
        ``stats.json``, and re-whitened so the returned tensor stays in the
        normalized space but is guaranteed feasible after unnormalization.

        Args:
            obs: The multimodal observation.
            clamp_feasible: If True, clamp to feasible joint limits via the
                normalizer's ``q01/q99``.

        Returns:
            Predicted action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        obs_tokens = self.encode_observation(obs)
        actions = self.action_head.sample_actions(obs_tokens)
        if clamp_feasible and self.action_normalizer is not None:
            # actions are (B, chunk, 65) whitened -> unnormalize, clamp, re-normalize
            # need to handle shape: normalizer expects (..., 65) last dim
            orig_shape = actions.shape
            flat = actions.reshape(-1, orig_shape[-1])
            # unnormalize to robot units, clamp to q01/q99, re-whiten
            # Use the normalizer's unnormalize_and_clamp then re-whiten via __call__
            # To avoid double, we do manual: flat_unnorm = flat * std + mean, clamp, then (clamped - mean)/std
            # The normalizer already does this in unnormalize_and_clamp + whiten
            unnorm = self.action_normalizer.unnormalize_and_clamp(flat, clamp="q01_q99")
            # re-whiten
            actions = self.action_normalizer(unnorm).reshape(orig_shape)
        return actions

    def sample_feasible_actions(self, obs: Observation) -> torch.Tensor:
        """Sample and return feasible actions in robot units.

        This is the user-facing inference API that always returns actions
        clamped to feasible limits (if a normalizer is available). The returned
        tensor is in original robot scale, not whitened.

        Args:
            obs: The multimodal observation.

        Returns:
            Feasible action chunk of shape ``(batch, chunk_size, action_dim)`` in
            original robot units.
        """
        whitened = self.sample_actions(obs, clamp_feasible=True)
        if self.action_normalizer is not None:
            # whitened is (B, chunk, 65) in whitened space, unnormalize to robot units
            # Use the normalizer's unnormalize (already clamped)
            B, C, D = whitened.shape
            flat = whitened.reshape(-1, D)
            feasible = self.action_normalizer.unnormalize(flat)
            return feasible.reshape(B, C, D)
        return whitened

    @override
    def forward(self, obs: Observation) -> torch.Tensor:
        """Sample actions, equivalent to :meth:`sample_actions`.

        Note:
            No ``torch.no_grad`` is applied internally, so ``model(obs)`` remains
            differentiable. Wrap the call in ``torch.inference_mode()`` at
            inference time.

        Args:
            obs: The multimodal observation.

        Returns:
            Predicted action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        return self.sample_actions(obs)
