"""Tensorclass types describing VLTA observations, inputs, and outputs."""

import numpy as np
import torch
from numpy import typing as npt
from tensordict import TensorClass as TensorClass

from origami_iros._typing import Image, TactileImage


class LeftRightImageObservation(TensorClass):
    """Images captured by a single stereo camera pair.

    Attributes:
        left: Image from the left camera of the pair.
        right: Image from the right camera of the pair.
    """

    left: Image
    right: Image


class RobotStateObservation(TensorClass):
    """Proprioceptive robot measurements accompanying an episode step.

    Attributes:
        joint_state: Arm joint measurements (e.g. positions and velocities).
        joint_torque: Torques measured at each arm joint.
        tactile: Low-dimensional tactile readings from the gripper sensors.
    """

    joint_state: torch.Tensor
    joint_torque: torch.Tensor
    tactile: torch.Tensor


class TactileImageObservation(TensorClass):
    """Tactile sensing data represented as images.

    Attributes:
        deform: Processed tactile image visualising contact deformation.
        raw: Unprocessed tactile image straight from the sensor.
    """

    deform: TactileImage
    raw: TactileImage


class ImageObservation(TensorClass):
    """All image-based modalities observed by the robot.

    Attributes:
        head: Images from the head-mounted stereo camera pair.
        wrist: Images from the wrist-mounted stereo camera pair.
        tactile: Images from the tactile sensors.
    """

    head: LeftRightImageObservation
    wrist: LeftRightImageObservation
    tactile: TactileImageObservation


class Observation(TensorClass):
    """Complete multimodal observation provided to the model at each step.

    Attributes:
        image: Camera and tactile image modalities.
        state: Proprioceptive joint and tactile measurements.
    """

    image: ImageObservation
    state: RobotStateObservation


class VLTA_Input(TensorClass):
    """Input to the vision-language-tactile-action (VLTA) policy.

    Attributes:
        observation: Multimodal robot observation for the current step.
        prompt: Natural-language instruction describing the task to perform.
    """

    observation: Observation
    prompt: str


class VLTA_Output(TensorClass):
    """Output produced by the vision-language-tactile-action (VLTA) policy.

    Attributes:
        action: Predicted action (e.g. an end-effector or joint command).
    """

    action: torch.Tensor


#: Flat dictionary representation of VLTA inputs and outputs, as produced by
#: :mod:`origami_iros.model.io`.
type DictData = dict[str, npt.NDArray[np.uint8 | np.float32] | str]
