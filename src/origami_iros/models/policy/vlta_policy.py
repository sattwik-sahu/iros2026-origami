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
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.action_head = action_head

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

    def sample_actions(self, obs: Observation) -> torch.Tensor:
        """Sample a chunk of actions for an observation.

        This is the differentiable prediction path: gradients flow back through
        the observation encoding and the velocity network. Inference callers
        should wrap the call in ``torch.inference_mode()`` / ``torch.no_grad()``
        explicitly (see :meth:`sample_for_logging` in the Lightning module).

        Args:
            obs: The multimodal observation.

        Returns:
            Predicted action chunk of shape ``(batch, chunk_size, action_dim)``.
        """
        obs_tokens = self.encode_observation(obs)
        return self.action_head.sample_actions(obs_tokens)

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
