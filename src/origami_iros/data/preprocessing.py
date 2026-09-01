"""Data preprocessors: normalization and image transforms.

Preprocessing sits between the raw Lerobot samples and the model. The project
derives its normalization statistics from the dataset's own ``meta/stats.json``
(so it is correct-by-construction and never drifts from the data), rather than
hardcoding per-feature constants.

Two kinds of preprocessing are provided:

* **Scalar normalization** (:class:`StatsNormalize`): zero-mean / unit-variance
  whitening of ``action``, ``observation.state``, ``observation.tactile`` and the
  joint-torque stream, using the mean/std stored in the dataset stats.
* **Image normalization** (:class:`ImageNormalize`): rescaling of raw tactile
  camera images. Camera views (head / wrist) are normalized inside the frozen HF
  image processor, so they are passed through unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from origami_iros.data.metadata import FeatureStats


@runtime_checkable
class Preprocessor(Protocol):
    """A callable that transforms a single dataset sample (or a value)."""

    def __call__(self, value: Any) -> Any:
        """Transform ``value``.

        Args:
            value: A raw value (e.g. a single feature tensor or a sample dict).

        Returns:
            The transformed value.
        """
        ...


class StatsNormalize:
    """Whiten a feature to zero mean / unit standard deviation.

    Attributes:
        feature_key: Dataset feature key that produced the statistics.
        eps: Small constant added to the standard deviation for numerical safety.
    """

    def __init__(self, feature_stats: FeatureStats, eps: float = 1e-8) -> None:
        """Initialise the normalizer.

        Args:
            feature_stats: Dataset statistics (mean/std) for the feature.
            eps: Small constant added to the standard deviation for numerical safety.
        """
        mean = np.asarray(feature_stats.mean, dtype=np.float32)
        std = np.asarray(feature_stats.std, dtype=np.float32)
        self.feature_key: str | None = None
        self._mean = torch.from_numpy(mean)
        self._std = torch.clamp(torch.from_numpy(std), min=eps)

    def __call__(self, value: Tensor) -> Tensor:
        """Whiten ``value``.

        Args:
            value: A tensor of shape ``(..., dim)`` matching the statistics.

        Returns:
            The whitened tensor.
        """
        if value.shape[-1] != self._mean.shape[-1]:
            raise ValueError(
                f"normalizer expects dim {self._mean.shape[-1]} but got {value.shape[-1]}"
            )
        return (value - self._mean) / self._std

    def unnormalize(self, value: Tensor) -> Tensor:
        """Invert the whitening transform.

        Args:
            value: A whitened tensor.

        Returns:
            The tensor in the original data scale.
        """
        return value * self._std + self._mean


class ImageNormalize:
    """Normalize an image tensor to a fixed linear range (default 0..1).

    Camera and tactile images leave the dataset already in the 0..1 range, so
    this is a near-identity preprocessor retained for symmetry and future
    per-modality scaling.

    Attributes:
        scale: Scaling applied to the image.
    """

    def __init__(self, scale: float = 1.0) -> None:
        """Initialise the preprocessor.

        Args:
            scale: Scaling factor.
        """
        self.scale = scale

    def __call__(self, value: Tensor) -> Tensor:
        """Apply the scaling.

        Args:
            value: A ``(c, h, w)`` image tensor.

        Returns:
            The scaled image.
        """
        return value * self.scale


class ObservationPreprocessor:
    """Apply per-feature preprocessors to a raw Lerobot sample.

    The raw sample is a dict keyed by feature name, e.g.
    ``{"observation.state": ..., "action": ..., ...}``. This class applies the
    registered preprocessors to their scoped keys and leaves everything else
    unchanged.

    Attributes:
        _normalizers: Mapping of feature key (or "observation.<key>") to normalizer.
        _image_keys: Camera/tactile image keys passed through the image normalizer.
    """

    def __init__(
        self,
        normalizers: Mapping[str, Preprocessor],
        image_keys: Sequence[str] = (),
        image_normalizer: Preprocessor | None = None,
    ) -> None:
        """Initialise.

        Args:
            normalizers: Mapping of feature keys to scalar preprocessors.
            image_keys: Feature keys treated as images.
            image_normalizer: Optional preprocessor applied to every image key.
        """
        self._normalizers = dict(normalizers)
        self._image_keys = set(image_keys)
        self._image_normalizer = image_normalizer

    @classmethod
    def from_dataset_stats(
        cls,
        stats,
        normalize_actions: bool = True,
        normalize_state: bool = True,
        normalize_tactile: bool = True,
        normalize_torque: bool = True,
        action_dim: int | None = None,
        state_dim: int | None = None,
    ) -> "ObservationPreprocessor":
        """Build the preprocessor from a :class:`DatasetStats` instance.

        One :class:`StatsNormalize` is created per selectable scalar modality
        using the corresponding entry of the dataset's ``stats.json``.

        Args:
            stats: The loaded :class:`DatasetStats`.
            normalize_actions: Whitening for the ``action`` feature.
            normalize_state: Whitening for ``observation.state``.
            normalize_tactile: Whitening for ``observation.tactile``.
            normalize_torque: Whitening for ``observation.state.joint_torque``.
            action_dim: Target action dimension (pads/trims stats if given).
            state_dim: Target state dimension (pads/trims stats if given).

        Returns:
            A configured :class:`ObservationPreprocessor`.
        """
        normalizers: dict[str, Preprocessor] = {}
        if normalize_actions:
            fs = stats.action(action_dim or 0)
            normalizers["action"] = StatsNormalize(fs)
        if normalize_state:
            fs = stats.state(state_dim or 0)
            normalizers["observation.state"] = StatsNormalize(fs)
        if normalize_tactile and "observation.tactile" in stats.by_feature:
            normalizers["observation.tactile"] = StatsNormalize(
                stats.anything("observation.tactile")
            )
        if normalize_torque and "observation.state.joint_torque" in stats.by_feature:
            normalizers["observation.state.joint_torque"] = StatsNormalize(
                stats.anything("observation.state.joint_torque")
            )
        return cls(
            normalizers=normalizers,
            image_keys=(
                "observation.images.head_left",
                "observation.images.head_right",
                "observation.images.wrist_left",
                "observation.images.wrist_right",
                "observation.images.tactile_raw",
                "observation.images.tactile_deform",
            ),
        )

    def __call__(self, sample: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return a new sample dict with the registered preprocessors applied."""
        out = dict(sample)
        for key, normalizer in self._normalizers.items():
            if key in out:
                out[key] = normalizer(out[key])
        if self._image_normalizer is not None:
            for key in self._image_keys:
                if key in out:
                    out[key] = self._image_normalizer(out[key])
        return out
