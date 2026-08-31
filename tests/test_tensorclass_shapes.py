"""Tests for VLTA TensorClass hierarchy — construction and tensor shapes."""

import torch

from origami_iros.modules._typing import (
    ImageObservation,
    LeftRightImageObservation,
    Observation,
    RobotStateObservation,
    TactileImageObservation,
    VLTA_Input,
    VLTA_Output,
)

BATCH = 2
C, H, W = 3, 224, 224
N_JOINTS = 7
N_TACTILE = 10


class TestLeftRightImageObservation:
    def test_construction(self, device):
        lr = LeftRightImageObservation(
            left=torch.randn(BATCH, C, H, W, device=device),
            right=torch.randn(BATCH, C, H, W, device=device),
        )
        assert lr.left.shape == (BATCH, C, H, W)
        assert lr.right.shape == (BATCH, C, H, W)

    def test_device(self, device):
        lr = LeftRightImageObservation(
            left=torch.randn(BATCH, C, H, W, device=device),
            right=torch.randn(BATCH, C, H, W, device=device),
        )
        assert lr.left.device.type == device.type
        assert lr.right.device.type == device.type


class TestTactileImageObservation:
    def test_construction(self, device):
        t = TactileImageObservation(
            deform=torch.randn(BATCH, C, H, W, device=device),
            raw=torch.randn(BATCH, C, H, W, device=device),
        )
        assert t.deform.shape == (BATCH, C, H, W)
        assert t.raw.shape == (BATCH, C, H, W)


class TestImageObservation:
    def test_construction(self, device):
        obs = ImageObservation(
            head=LeftRightImageObservation(
                left=torch.randn(BATCH, C, H, W, device=device),
                right=torch.randn(BATCH, C, H, W, device=device),
            ),
            wrist=LeftRightImageObservation(
                left=torch.randn(BATCH, C, H, W, device=device),
                right=torch.randn(BATCH, C, H, W, device=device),
            ),
            tactile=TactileImageObservation(
                deform=torch.randn(BATCH, C, H, W, device=device),
                raw=torch.randn(BATCH, C, H, W, device=device),
            ),
        )
        assert obs.head.left.shape == (BATCH, C, H, W)
        assert obs.wrist.right.shape == (BATCH, C, H, W)
        assert obs.tactile.raw.shape == (BATCH, C, H, W)

    def test_nested_access(self, device):
        imgs = {
            "head_left": torch.randn(BATCH, C, H, W, device=device),
            "head_right": torch.randn(BATCH, C, H, W, device=device),
            "wrist_left": torch.randn(BATCH, C, H, W, device=device),
            "wrist_right": torch.randn(BATCH, C, H, W, device=device),
            "tactile_deform": torch.randn(BATCH, C, H, W, device=device),
            "tactile_raw": torch.randn(BATCH, C, H, W, device=device),
        }
        obs = ImageObservation(
            head=LeftRightImageObservation(left=imgs["head_left"], right=imgs["head_right"]),
            wrist=LeftRightImageObservation(left=imgs["wrist_left"], right=imgs["wrist_right"]),
            tactile=TactileImageObservation(deform=imgs["tactile_deform"], raw=imgs["tactile_raw"]),
        )
        assert obs.head.left is imgs["head_left"]
        assert obs.tactile.raw is imgs["tactile_raw"]


class TestRobotStateObservation:
    def test_construction(self, device):
        rs = RobotStateObservation(
            joint_state=torch.randn(BATCH, N_JOINTS, device=device),
            joint_torque=torch.randn(BATCH, N_JOINTS, device=device),
            tactile=torch.randn(BATCH, N_TACTILE, device=device),
        )
        assert rs.joint_state.shape == (BATCH, N_JOINTS)
        assert rs.joint_torque.shape == (BATCH, N_JOINTS)
        assert rs.tactile.shape == (BATCH, N_TACTILE)

    def test_single_sample(self, device):
        rs = RobotStateObservation(
            joint_state=torch.randn(1, N_JOINTS, device=device),
            joint_torque=torch.randn(1, N_JOINTS, device=device),
            tactile=torch.randn(1, N_TACTILE, device=device),
        )
        assert rs.joint_state.shape == (1, N_JOINTS)


class TestObservation:
    def test_construction(self, device):
        obs = Observation(
            image=ImageObservation(
                head=LeftRightImageObservation(
                    left=torch.randn(BATCH, C, H, W, device=device),
                    right=torch.randn(BATCH, C, H, W, device=device),
                ),
                wrist=LeftRightImageObservation(
                    left=torch.randn(BATCH, C, H, W, device=device),
                    right=torch.randn(BATCH, C, H, W, device=device),
                ),
                tactile=TactileImageObservation(
                    deform=torch.randn(BATCH, C, H, W, device=device),
                    raw=torch.randn(BATCH, C, H, W, device=device),
                ),
            ),
            state=RobotStateObservation(
                joint_state=torch.randn(BATCH, N_JOINTS, device=device),
                joint_torque=torch.randn(BATCH, N_JOINTS, device=device),
                tactile=torch.randn(BATCH, N_TACTILE, device=device),
            ),
        )
        assert obs.image.head.left.shape == (BATCH, C, H, W)
        assert obs.state.joint_state.shape == (BATCH, N_JOINTS)


class TestVLTAInput:
    def test_construction(self, observation, device):
        vi = VLTA_Input(observation=observation, prompt="pick up the cup")
        assert vi.prompt == "pick up the cup"
        assert vi.observation.image.head.left.shape == (BATCH, C, H, W)

    def test_batch_shapes_preserved(self, device):
        for batch in [1, 4, 8]:
            obs = Observation(
                image=ImageObservation(
                    head=LeftRightImageObservation(
                        left=torch.randn(batch, C, H, W, device=device),
                        right=torch.randn(batch, C, H, W, device=device),
                    ),
                    wrist=LeftRightImageObservation(
                        left=torch.randn(batch, C, H, W, device=device),
                        right=torch.randn(batch, C, H, W, device=device),
                    ),
                    tactile=TactileImageObservation(
                        deform=torch.randn(batch, C, H, W, device=device),
                        raw=torch.randn(batch, C, H, W, device=device),
                    ),
                ),
                state=RobotStateObservation(
                    joint_state=torch.randn(batch, N_JOINTS, device=device),
                    joint_torque=torch.randn(batch, N_JOINTS, device=device),
                    tactile=torch.randn(batch, N_TACTILE, device=device),
                ),
            )
            vi = VLTA_Input(observation=obs, prompt=f"task_{batch}")
            assert vi.observation.image.head.left.shape[0] == batch
            assert vi.observation.state.joint_state.shape[0] == batch


class TestVLTAOutput:
    def test_construction(self, device):
        vo = VLTA_Output(action=torch.randn(BATCH, 7, device=device))
        assert vo.action.shape == (BATCH, 7)

    def test_various_action_dims(self, device):
        for dim in [4, 7, 14]:
            vo = VLTA_Output(action=torch.randn(BATCH, dim, device=device))
            assert vo.action.shape == (BATCH, dim)
