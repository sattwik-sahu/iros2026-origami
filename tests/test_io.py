"""Tests for VLTA I/O serialization roundtrips."""

import numpy as np
import pytest
import torch

from origami_iros.models._typing import VLTA_Input, VLTA_Output
from origami_iros.data.io import (
    to_vlta_input,
    to_vlta_output,
    vlta_input_to_dict,
    vlta_output_to_dict,
)

BATCH = 2
C, H, W = 3, 224, 224
N_JOINTS = 7
N_TACTILE = 10
DIM_ACTION = 7


class TestVLTAInputRoundtrip:
    def test_roundtrip_shapes(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        reconstructed = to_vlta_input(d)

        assert reconstructed.observation.image.head.left.shape == (BATCH, C, H, W)
        assert reconstructed.observation.image.head.right.shape == (BATCH, C, H, W)
        assert reconstructed.observation.image.wrist.left.shape == (BATCH, C, H, W)
        assert reconstructed.observation.image.wrist.right.shape == (BATCH, C, H, W)
        assert reconstructed.observation.image.tactile.deform.shape == (BATCH, C, H, W)
        assert reconstructed.observation.image.tactile.raw.shape == (BATCH, C, H, W)
        assert reconstructed.observation.state.joint_state.shape == (BATCH, N_JOINTS)
        assert reconstructed.observation.state.joint_torque.shape == (BATCH, N_JOINTS)
        assert reconstructed.observation.state.tactile.shape == (BATCH, N_TACTILE)

    def test_roundtrip_values(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        reconstructed = to_vlta_input(d)

        torch.testing.assert_close(
            reconstructed.observation.image.head.left,
            vlta_input.observation.image.head.left.cpu(),
        )
        torch.testing.assert_close(
            reconstructed.observation.state.joint_state,
            vlta_input.observation.state.joint_state.cpu(),
        )

    def test_roundtrip_prompt(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        reconstructed = to_vlta_input(d)
        assert reconstructed.prompt == vlta_input.prompt

    def test_dict_keys(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        expected_keys = {
            "observation/image/head_left",
            "observation/image/head_right",
            "observation/image/wrist_left",
            "observation/image/wrist_right",
            "observation/state",
            "observation/state/joint_torque",
            "observation/tactile",
            "observation/image/tactile_deform",
            "observation/image/tactile_raw",
            "prompt",
        }
        assert set(d.keys()) == expected_keys

    def test_dict_values_are_numpy(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        for key in d:
            if key == "prompt":
                assert isinstance(d[key], str)
            else:
                assert isinstance(d[key], np.ndarray)

    def test_dict_numpy_dtypes(self, vlta_input):
        d = vlta_input_to_dict(vlta_input)
        for key, val in d.items():
            if key == "prompt":
                continue
            assert val.dtype in (np.float32, np.uint8), f"{key} has dtype {val.dtype}"


class TestVLTAOutputRoundtrip:
    def test_roundtrip_shape(self, vlta_output):
        d = vlta_output_to_dict(vlta_output)
        reconstructed = to_vlta_output(d)
        assert reconstructed.action.shape == (BATCH, DIM_ACTION)

    def test_roundtrip_values(self, vlta_output):
        d = vlta_output_to_dict(vlta_output)
        reconstructed = to_vlta_output(d)
        torch.testing.assert_close(reconstructed.action, vlta_output.action.cpu())

    def test_dict_key(self, vlta_output):
        d = vlta_output_to_dict(vlta_output)
        assert set(d.keys()) == {"action"}

    def test_numpy_dtype(self, vlta_output):
        d = vlta_output_to_dict(vlta_output)
        assert isinstance(d["action"], np.ndarray)


class TestErrorHandling:
    def test_to_vlta_input_missing_key(self):
        with pytest.raises(KeyError):
            to_vlta_input({"prompt": "hello"})

    def test_to_vlta_input_string_for_tensor_key(self):
        bad_dict = {
            "observation/image/head_left": "not_an_array",
            "observation/image/head_right": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "observation/image/wrist_left": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "observation/image/wrist_right": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "observation/state": np.zeros((1, 7), dtype=np.float32),
            "observation/state/joint_torque": np.zeros((1, 7), dtype=np.float32),
            "observation/tactile": np.zeros((1, 10), dtype=np.float32),
            "observation/image/tactile_deform": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "observation/image/tactile_raw": np.zeros((1, 3, 224, 224), dtype=np.float32),
            "prompt": "test",
        }
        with pytest.raises(TypeError, match="expected a numpy array"):
            to_vlta_input(bad_dict)

    def test_to_vlta_output_string_for_action(self):
        with pytest.raises(TypeError, match="expected a numpy array"):
            to_vlta_output({"action": "not_an_array"})

    def test_to_vlta_output_missing_action(self):
        with pytest.raises(KeyError):
            to_vlta_output({})
