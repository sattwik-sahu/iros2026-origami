# origami_iros/modules/data/collate.py
import torch
from torch.utils.data import default_collate

from origami_iros.modules._typing import (
    Observation,
    ImageObservation,
    LeftRightImageObservation,
    TactileImageObservation,
    RobotStateObservation,
)


def build_observation(batch: dict) -> Observation:
    n = batch["observation.state"].shape[0]
    return Observation(
        image=ImageObservation(
            head=LeftRightImageObservation(
                left=batch["observation.images.head_left"],
                right=batch["observation.images.head_right"],
            ),
            wrist=LeftRightImageObservation(
                left=batch["observation.images.wrist_left"],
                right=batch["observation.images.wrist_right"],
            ),
            tactile=TactileImageObservation(
                deform=batch["observation.images.tactile_deform"],
                raw=batch["observation.images.tactile_raw"],
            ),
            batch_size=[n],
        ),
        state=RobotStateObservation(
            joint_state=batch["observation.state"],
            joint_torque=batch["observation.state.joint_torque"],
            tactile=batch["observation.tactile"],
            batch_size=[n],
        ),
        batch_size=[n],
    )


def vlta_collate_fn(samples: list[dict]):
    batch = default_collate(samples)
    obs = build_observation(batch)
    action = batch["action"].reshape(batch["action"].shape[0], -1)
    action_is_pad = batch.get("action_is_pad")
    return obs, action, action_is_pad