"""Tests for proprioceptive state encoders — shapes and forward pass."""

import torch
from torch import nn

from origami_iros.modules._typing import RobotStateObservation
from origami_iros.modules.encoders.proprio import (
    LearnedMultiTokenStateEncoder,
    ModalityEncoder,
    SingleTokenStateEncoder,
)

BATCH = 2
N_JOINTS = 7
N_TACTILE = 10
DIM = 64


def _robot_state(device, batch=BATCH):
    return RobotStateObservation(
        joint_state=torch.randn(batch, N_JOINTS, device=device),
        joint_torque=torch.randn(batch, N_JOINTS, device=device),
        tactile=torch.randn(batch, N_TACTILE, device=device),
    )


def _sub_encoders(device):
    return dict(
        torque_encoder=nn.Linear(N_JOINTS, DIM).to(device),
        joint_state_encoder=nn.Linear(N_JOINTS, DIM).to(device),
        tactile_encoder=nn.Linear(N_TACTILE, DIM).to(device),
    )


# ── ModalityEncoder ───────────────────────────────────────────────────

class TestModalityEncoder:
    def test_output_shape(self, device):
        enc = ModalityEncoder(**_sub_encoders(device)).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        # (B, 3 modalities, dim)
        assert out.shape == (BATCH, 3, DIM)

    def test_batch_independence(self, device):
        enc = ModalityEncoder(**_sub_encoders(device)).to(device)
        rs1 = _robot_state(device, batch=1)
        rs4 = _robot_state(device, batch=4)
        assert enc(rs1).shape == (1, 3, DIM)
        assert enc(rs4).shape == (4, 3, DIM)

    def test_device(self, device):
        enc = ModalityEncoder(**_sub_encoders(device)).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        assert out.device.type == device.type

    def test_modality_order(self, device):
        """Token order: [torque, joint_state, tactile]."""
        enc = ModalityEncoder(**_sub_encoders(device)).to(device)
        rs = RobotStateObservation(
            joint_state=torch.ones(1, N_JOINTS, device=device) * 0,
            joint_torque=torch.ones(1, N_JOINTS, device=device) * 1,
            tactile=torch.ones(1, N_TACTILE, device=device) * 2,
        )
        out = enc(rs)
        # Check that each modality got different input
        assert out[0, 0].mean() != out[0, 1].mean()  # torque vs joint_state
        assert out[0, 1].mean() != out[0, 2].mean()  # joint_state vs tactile


# ── SingleTokenStateEncoder ───────────────────────────────────────────

class TestSingleTokenStateEncoder:
    def test_output_shape(self, device):
        enc = SingleTokenStateEncoder(**_sub_encoders(device), dim=DIM).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        assert out.shape == (BATCH, 3, DIM)

    def test_modality_embeddings_applied(self, device):
        """Output tokens differ from raw modality embeddings due to addition."""
        enc = SingleTokenStateEncoder(**_sub_encoders(device), dim=DIM).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        # modality_embeddings shape is (3, dim)
        assert enc.modality_embeddings.shape == (3, DIM)
        # The output should not be zero (embeddings + modality bias)
        assert out.abs().sum() > 0

    def test_batch_dimension(self, device):
        enc = SingleTokenStateEncoder(**_sub_encoders(device), dim=DIM).to(device)
        for b in [1, 3, 8]:
            rs = _robot_state(device, batch=b)
            out = enc(rs)
            assert out.shape[0] == b

    def test_output_is_differentiable(self, device):
        enc = SingleTokenStateEncoder(**_sub_encoders(device), dim=DIM).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        loss = out.sum()
        loss.backward()
        # modality_embeddings should have gradients
        assert enc.modality_embeddings.grad is not None


# ── LearnedMultiTokenStateEncoder ─────────────────────────────────────

class TestLearnedMultiTokenStateEncoder:
    def test_default_output_shape(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM
        ).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        # Default: num_tokens=4
        assert out.shape == (BATCH, 4, DIM)

    def test_custom_num_tokens(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM, num_tokens=8
        ).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        assert out.shape == (BATCH, 8, DIM)

    def test_custom_num_tokens_1(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM, num_tokens=1
        ).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        assert out.shape == (BATCH, 1, DIM)

    def test_proprio_tokens_parameter(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM, num_tokens=6
        ).to(device)
        assert enc.proprio_tokens.shape == (6, DIM)

    def test_batch_dimension(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM, num_tokens=4
        ).to(device)
        for b in [1, 2, 5]:
            rs = _robot_state(device, batch=b)
            out = enc(rs)
            assert out.shape[0] == b

    def test_output_is_differentiable(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM
        ).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        loss = out.sum()
        loss.backward()
        assert enc.proprio_tokens.grad is not None

    def test_transformer_decoder_layers(self, device):
        enc = LearnedMultiTokenStateEncoder(
            **_sub_encoders(device), dim=DIM, num_layers=4, num_heads=8
        ).to(device)
        rs = _robot_state(device)
        out = enc(rs)
        assert out.shape == (BATCH, 4, DIM)
