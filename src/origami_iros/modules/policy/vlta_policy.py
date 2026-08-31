# origami_iros/modules/policy/vlta_policy.py
import torch
from torch import nn

from origami_iros.modules._typing import Observation, RobotStateObservation
from origami_iros.modules.encoders.image import (
    CameraImageEncoder,
    TactileImageEncoder,
    PretrainedHF_ViT_Encoder,
    TinyViT_TactileImageEncoder,
)
from origami_iros.modules.encoders.proprio import SingleTokenStateEncoder
from origami_iros.modules.action_head.fm import FlowMatchingActionHead


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLTAPolicy(nn.Module):
    def __init__(
        self,
        vit_model_name: str,
        image_size: tuple[int, int],
        vit_dim: int,
        tactile_image_size: tuple[int, int],
        tactile_patch_size: int,
        tactile_dim: int,
        n_hands: int,
        n_fingers: int,
        torque_dim: int,
        joint_state_dim: int,
        proprio_tactile_dim: int,
        hidden_dim: int,
        chunk_size: int,
        action_dim: int,
        action_hidden_dim: int,
        action_num_layers: int,
        action_num_heads: int,
        num_inference_steps: int,
        freeze_vit: bool = True,
    ) -> None:
        super().__init__()

        vit = PretrainedHF_ViT_Encoder(image_size=image_size, model_name=vit_model_name)
        if freeze_vit:
            for p in vit.parameters():
                p.requires_grad_(False)
        self.camera_encoder = CameraImageEncoder(encoder=vit)
        self.vit_out_proj = nn.Linear(vit_dim, hidden_dim)

        deform_vit = TinyViT_TactileImageEncoder(image_size=(480, 1200), patch_size=tactile_patch_size, n_hands=2, n_fingers=5)
        raw_vit = TinyViT_TactileImageEncoder(image_size=(480, 1600), patch_size=tactile_patch_size, n_hands=2, n_fingers=5)
        self.tactile_encoder = TactileImageEncoder(deform_encoder=deform_vit, raw_encoder=raw_vit)
        self.tactile_out_proj = nn.Linear(tactile_dim, hidden_dim)

        self.proprio_encoder = SingleTokenStateEncoder(
            torque_encoder=MLPEncoder(torque_dim, hidden_dim),
            joint_state_encoder=MLPEncoder(joint_state_dim, hidden_dim),
            tactile_encoder=MLPEncoder(proprio_tactile_dim, hidden_dim),
            dim=hidden_dim,
        )

        self.action_head = FlowMatchingActionHead(
            chunk_size=chunk_size,
            action_dim=action_dim,
            dim_in=hidden_dim,
            hidden_dim=action_hidden_dim,
            num_layers=action_num_layers,
            num_heads=action_num_heads,
            num_inference_steps=num_inference_steps,
        )

    def encode_observation(self, obs: Observation) -> torch.Tensor:
        image_tokens = self.vit_out_proj(self.camera_encoder(obs.image))
        tactile_tokens = self.tactile_out_proj(self.tactile_encoder(obs.image))
        proprio_tokens = self.proprio_encoder(obs.state)
        return torch.cat([image_tokens, tactile_tokens, proprio_tokens], dim=1)

    def compute_loss(self, obs: Observation, target_action: torch.Tensor) -> torch.Tensor:
        obs_tokens = self.encode_observation(obs)
        return self.action_head.compute_loss(obs_tokens, target_action)

    @torch.no_grad()
    def sample_actions(self, obs: Observation) -> torch.Tensor:
        obs_tokens = self.encode_observation(obs)
        return self.action_head.sample_actions(obs_tokens)