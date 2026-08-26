"""Serialization utilities for VLTA inputs and outputs.

These helpers convert between the structured
:class:`~origami_iros.modules._typing.VLTA_Input` /
:class:`~origami_iros.modules._typing.VLTA_Output` tensorclasses and a flat
:class:`~origami_iros.modules._typing.DictData` dictionary of numpy arrays and
strings, which is a convenient format for logging, caching, or sending data
across process boundaries.
"""

import torch

from origami_iros.modules._typing import (
    DictData,
    ImageObservation,
    LeftRightImageObservation,
    Observation,
    RobotStateObservation,
    TactileImageObservation,
)
from origami_iros.modules._typing import VLTA_Input as VLTA_Input
from origami_iros.modules._typing import VLTA_Output as VLTA_Output


def _to_tensor(d: DictData, key: str) -> torch.Tensor:
    """Return ``d[key]`` as a CPU tensor, raising if the entry is a string."""
    value = d[key]
    if isinstance(value, str):
        msg = f"expected a numpy array for key {key!r}, got a string"
        raise TypeError(msg)
    return torch.from_numpy(value)


def _to_str(d: DictData, key: str) -> str:
    """Return ``d[key]`` as a string, raising if the entry is an array."""
    value = d[key]
    if not isinstance(value, str):
        msg = f"expected a string for key {key!r}, got a numpy array"
        raise TypeError(msg)
    return value


def vlta_input_to_dict(v: VLTA_Input) -> DictData:
    """Flatten a VLTA input into a dictionary of numpy arrays and strings.

    Every tensor is moved to the CPU before conversion; dtypes and shapes
    are preserved.

    Args:
        v: Structured model input to flatten.

    Returns:
        Dictionary mapping dotted keys such as
        ``"observation/image/head_left"`` to numpy arrays, plus the task
        instruction under the ``"prompt"`` key.

    See Also:
        :func:`to_vlta_input`: the inverse transformation.
    """
    return {
        "observation/image/head_left": v.observation.image.head.left.cpu().numpy(),
        "observation/image/head_right": v.observation.image.head.right.cpu().numpy(),
        "observation/image/wrist_left": v.observation.image.wrist.left.cpu().numpy(),
        "observation/image/wrist_right": v.observation.image.wrist.right.cpu().numpy(),
        "observation/state": v.observation.state.joint_state.cpu().numpy(),
        "observation/state/joint_torque": v.observation.state.joint_torque.cpu().numpy(),
        "observation/tactile": v.observation.state.tactile.cpu().numpy(),
        "observation/image/tactile_deform": v.observation.image.tactile.deform.cpu().numpy(),
        "observation/image/tactile_raw": v.observation.image.tactile.raw.cpu().numpy(),
        "prompt": v.prompt,
    }


def vlta_output_to_dict(v: VLTA_Output) -> DictData:
    """Flatten a VLTA output into a dictionary of numpy arrays.

    Args:
        v: Structured model output to flatten. The action tensor is moved to
            the CPU before conversion.

    Returns:
        Dictionary holding the predicted action(s) under the ``"action"``
        key.

    See Also:
        :func:`to_vlta_output`: the inverse transformation.
    """
    return {"action": v.action.cpu().numpy()}


def to_vlta_input(d: DictData) -> VLTA_Input:
    """Rebuild a VLTA input tensorclass from its flattened representation.

    This is the inverse of :func:`vlta_input_to_dict`. The resulting tensors
    live on the CPU and share memory with the underlying numpy arrays.

    Args:
        d: Flattened dictionary of numpy arrays and strings, as produced by
            :func:`vlta_input_to_dict`.

    Returns:
        The reconstructed structured model input.

    Raises:
        KeyError: If ``d`` is missing one of the expected entries.
        TypeError: If an array-valued key holds a string, or vice versa.
    """
    return VLTA_Input(
        observation=Observation(
            image=ImageObservation(
                head=LeftRightImageObservation(
                    left=_to_tensor(d, "observation/image/head_left"),
                    right=_to_tensor(d, "observation/image/head_right"),
                ),
                wrist=LeftRightImageObservation(
                    left=_to_tensor(d, "observation/image/wrist_left"),
                    right=_to_tensor(d, "observation/image/wrist_right"),
                ),
                tactile=TactileImageObservation(
                    deform=_to_tensor(d, "observation/image/tactile_deform"),
                    raw=_to_tensor(d, "observation/image/tactile_raw"),
                ),
            ),
            state=RobotStateObservation(
                joint_state=_to_tensor(d, "observation/state"),
                joint_torque=_to_tensor(d, "observation/state/joint_torque"),
                tactile=_to_tensor(d, "observation/tactile"),
            ),
        ),
        prompt=_to_str(d, "prompt"),
    )


def to_vlta_output(d: DictData) -> VLTA_Output:
    """Rebuild a VLTA output tensorclass from its flattened representation.

    This is the inverse of :func:`vlta_output_to_dict`. The resulting action
    tensor lives on the CPU and shares memory with the underlying numpy
    array.

    Args:
        d: Flattened dictionary containing an ``"action"`` entry, as
            produced by :func:`vlta_output_to_dict`.

    Returns:
        The reconstructed structured model output.

    Raises:
        KeyError: If ``d`` has no ``"action"`` entry.
        TypeError: If the ``"action"`` entry is a string rather than an array.
    """
    return VLTA_Output(action=_to_tensor(d, "action"))
