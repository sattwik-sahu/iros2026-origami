"""Hydra ``_target_`` constructors for the VLTA policy.

The policy itself (:class:`VLTAPolicy`) only wires together an already-built
observation encoder and an action head. This module provides the functions that
Hydra invokes via ``_target_`` to actually build those two pieces from the
(resolved) :class:`ModelConfig`. Keeping the assembly here means the policy class
stays free of hardcoded encoder/action-head internals, and the whole model
composition is declaratively described in the Hydra config.
"""

from __future__ import annotations

import torch
from typing import override
from torch import nn

from origami_iros.models.action_head.fm import FlowMatchingActionHead
from origami_iros.models.encoders.image import (
    CameraImageEncoder,
    PretrainedHF_ViT_Encoder,
    TactileImageEncoder,
    TinyViT_TactileImageEncoder,
)
from origami_iros.models.encoders.main import VLTA_Encoder
from origami_iros.models.encoders.proprio import SingleTokenStateEncoder
from origami_iros.models.policy.vlta_policy import VLTAPolicy
from origami_iros.train.config import ModelConfig


class MLPEncoder(nn.Module):
    """A small multi-layer perceptron used to encode a state modality."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        """Initialise the MLP.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            hidden_dim: Width of the single hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Args:
            x: Input tensor of shape ``(batch, in_dim)``.

        Returns:
            Output tensor of shape ``(batch, out_dim)``.
        """
        return self.net(x)


def build_vlta_encoder(cfg: ModelConfig) -> VLTA_Encoder:
    """Construct the observation encoder from a resolved model config.

    Args:
        cfg: The resolved :class:`ModelConfig` (data facts already filled in).

    Returns:
        A :class:`VLTA_Encoder` ready for use by the policy.
    """
    vit = PretrainedHF_ViT_Encoder(image_size=cfg.image_size, model_name=cfg.vit_model_name)
    if cfg.freeze_vit:
        for p in vit.parameters():
            p.requires_grad_(False)
    camera_image_encoder = CameraImageEncoder(encoder=vit)

    raw_vit = TinyViT_TactileImageEncoder(
        image_size=cfg.tactile_image_size,
        patch_size=cfg.tactile_patch_size,
        n_hands=cfg.n_hands,
        n_fingers=cfg.n_fingers,
    )
    # The recorded dataset's deform stream is all zeros, so the raw tactile
    # image is the primary signal. A secondary (deform) encoder can be wired in
    # later once the recording pipeline produces valid deform frames.
    tactile_image_encoder = TactileImageEncoder(primary_encoder=raw_vit, primary_key="raw")

    state_encoder = SingleTokenStateEncoder(
        torque_encoder=MLPEncoder(cfg.torque_dim, cfg.hidden_dim),
        joint_state_encoder=MLPEncoder(cfg.joint_state_dim, cfg.hidden_dim),
        tactile_encoder=MLPEncoder(cfg.proprio_tactile_dim, cfg.hidden_dim),
        dim=cfg.hidden_dim,
    )

    return VLTA_Encoder(
        camera_image_encoder=camera_image_encoder,
        tactile_image_encoder=tactile_image_encoder,
        state_encoder=state_encoder,
    )


def build_action_head(cfg: ModelConfig, chunk_size: int) -> FlowMatchingActionHead:
    """Construct the flow-matching action head from a resolved model config.

    Args:
        cfg: The resolved :class:`ModelConfig`.
        chunk_size: Number of action timesteps sampled per chunk.

    Returns:
        A :class:`FlowMatchingActionHead`.
    """
    return FlowMatchingActionHead(
        chunk_size=chunk_size,
        action_dim=cfg.action_dim,
        dim_in=cfg.hidden_dim,
        hidden_dim=cfg.action_hidden_dim,
        num_layers=cfg.action_num_layers,
        num_heads=cfg.action_num_heads,
        num_inference_steps=cfg.num_inference_steps,
    )


def build_vlta_policy(
    cfg: ModelConfig, chunk_size: int, action_normalizer: Any | None = None
) -> VLTAPolicy:
    """Assemble the full policy from a resolved model config.

    This is the Hydra ``_target_`` factory: it builds the observation encoder and
    the action head, then wires them into a :class:`VLTAPolicy`.

    Args:
        cfg: The resolved :class:`ModelConfig` (data facts already filled in).
        chunk_size: Number of action timesteps sampled per chunk.
        action_normalizer: Optional normalizer for feasible clamping at inference.
            When provided, ``sample_actions(..., clamp_feasible=True)`` will
            guarantee outputs are within ``q01/q99`` limits.

    Returns:
        A fully-constructed :class:`VLTAPolicy`.
    """
    encoder = build_vlta_encoder(cfg)
    action_head = build_action_head(cfg, chunk_size)
    return VLTAPolicy(
        encoder=encoder,
        action_head=action_head,
        vit_dim=cfg.vit_dim,
        tactile_dim=cfg.tactile_dim,
        hidden_dim=cfg.hidden_dim,
        action_normalizer=action_normalizer,
    )
