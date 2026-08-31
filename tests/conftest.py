import pytest
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
IMG_SIZE = (224, 224)
N_JOINTS = 7
N_TACTILE = 10
N_CHANNELS = 3
DIM_EMBED = 64
DIM_ACTION = 7


@pytest.fixture()
def device():
    dev = torch.accelerator.current_accelerator()
    return dev


@pytest.fixture()
def robot_state(device):
    return RobotStateObservation(
        joint_state=torch.randn(BATCH, N_JOINTS, device=device),
        joint_torque=torch.randn(BATCH, N_JOINTS, device=device),
        tactile=torch.randn(BATCH, N_TACTILE, device=device),
    )


@pytest.fixture()
def image_observation(device):
    return ImageObservation(
        head=LeftRightImageObservation(
            left=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
            right=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
        ),
        wrist=LeftRightImageObservation(
            left=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
            right=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
        ),
        tactile=TactileImageObservation(
            deform=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
            raw=torch.randn(BATCH, N_CHANNELS, *IMG_SIZE, device=device),
        ),
    )


@pytest.fixture()
def observation(image_observation, robot_state):
    return Observation(image=image_observation, state=robot_state)


@pytest.fixture()
def vlta_input(observation):
    return VLTA_Input(observation=observation, prompt="pick up the red cube")


@pytest.fixture()
def vlta_output(device):
    return VLTA_Output(action=torch.randn(BATCH, DIM_ACTION, device=device))
