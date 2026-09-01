import torch
from typing import override
from torch import nn

from origami_iros.models._typing import RobotStateObservation
from origami_iros.models.base import BaseProprioceptiveStateEncoder


class ModalityEncoder(nn.Module):
    """Encode proprioceptive modalities independently."""

    def __init__(
        self,
        torque_encoder: nn.Module,
        joint_state_encoder: nn.Module,
        tactile_encoder: nn.Module,
    ) -> None:
        super().__init__()

        self.torque_encoder = torque_encoder
        self.joint_state_encoder = joint_state_encoder
        self.tactile_encoder = tactile_encoder

    @override
    def forward(self, x: RobotStateObservation) -> torch.Tensor:
        return torch.stack(
            [
                self.torque_encoder(x.joint_torque),
                self.joint_state_encoder(x.joint_state),
                self.tactile_encoder(x.tactile),
            ],
            dim=1,
        )


class SingleTokenStateEncoder(BaseProprioceptiveStateEncoder[torch.Tensor]):
    """One token per proprioceptive modality."""

    def __init__(
        self,
        torque_encoder: nn.Module,
        joint_state_encoder: nn.Module,
        tactile_encoder: nn.Module,
        dim: int,
    ) -> None:
        super().__init__()

        self.encoder = ModalityEncoder(
            torque_encoder=torque_encoder,
            joint_state_encoder=joint_state_encoder,
            tactile_encoder=tactile_encoder,
        )

        self.modality_embeddings = nn.Parameter(torch.randn(3, dim))

    @override
    def forward(self, x: RobotStateObservation) -> torch.Tensor:
        tokens = self.encoder(x)
        return tokens + self.modality_embeddings.unsqueeze(0)


class LearnedMultiTokenStateEncoder(BaseProprioceptiveStateEncoder[torch.Tensor]):
    """Encode proprioception into learned latent tokens."""

    def __init__(
        self,
        torque_encoder: nn.Module,
        joint_state_encoder: nn.Module,
        tactile_encoder: nn.Module,
        dim: int,
        num_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.encoder = ModalityEncoder(
            torque_encoder=torque_encoder,
            joint_state_encoder=joint_state_encoder,
            tactile_encoder=tactile_encoder,
        )

        self.proprio_tokens = nn.Parameter(torch.randn(num_tokens, dim))

        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.tokenizer = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
        )

    @override
    def forward(self, x: RobotStateObservation) -> torch.Tensor:
        modality_tokens = self.encoder(x)

        queries = self.proprio_tokens.unsqueeze(0).expand(
            modality_tokens.shape[0],
            -1,
            -1,
        )

        return self.tokenizer(
            tgt=queries,
            memory=modality_tokens,
        )
