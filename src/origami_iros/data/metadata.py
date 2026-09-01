"""Dataset metadata discovery and typed accessors.

Lerobot v3.0 datasets ship a self-describing ``meta/`` folder per season:
``info.json`` (shapes, fps, features), ``modality.json`` (modality segmentation),
``stats.json`` (per-feature statistics used for normalization) and ``tasks.parquet``
(episode-level task annotations).

Rather than hardcoding dataset parameters (fps, action dimension, image sizes,
tactile dimension, ...) in the training configuration and risking silent drift,
the module below reads them from ``info.json`` at runtime. This keeps the Hydra
config minimal and correct-by-construction with respect to the data.

Only a single season is inspected; all seasons of the same project are expected to
share the same feature layout (which is enforced when the training datasets are
built).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


def _to_int_list(value: Any) -> list[int]:
    """Normalise ``value`` (Hamming for both ints and wrapped lists) to a list of ints."""
    return [int(v) for v in list(value)]


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from the dataset ``meta`` folder."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class DatasetMetadata:
    """Runtime-read metadata for a single Lerobot season.

    Attributes:
        fps: Frames per second of the dataset.
        action_dim: Dimension of the ``action`` feature.
        state_dim: Dimension of the ``observation.state`` feature.
        joint_torque_dim: Dimension of ``observation.state.joint_torque``.
        tcp_dim: Dimension of ``observation.state.tcp``.
        tactile_dim: Dimension of the ``observation.tactile`` state vector.
        state_names: Feature name list for ``action`` (aligned with ``state_dim``).
    """

    fps: float
    action_dim: int
    state_dim: int
    joint_torque_dim: int
    tcp_dim: int
    tactile_dim: int
    action_names: list[str] = field(default_factory=list)
    camera_size: tuple[int, int] = (0, 0)
    tactile_raw_size: tuple[int, int] = (0, 0)
    tactile_deform_size: tuple[int, int] = (0, 0)

    @classmethod
    def from_info(cls, info: dict[str, Any], modality: dict[str, Any]) -> "DatasetMetadata":
        """Build metadata from the parsed ``info.json`` and ``modality.json``.

        Args:
            info: Parsed contents of ``meta/info.json``.
            modality: Parsed contents of ``meta/modality.json``.

        Returns:
            A populated :class:`DatasetMetadata` instance.

        Raises:
            KeyError: If a required feature is missing from the metadata.
        """
        features = info["features"]
        joint_feature = modality.get("state", {}).get("joints", {})
        joint_key = joint_feature.get("original_key", "observation.state")

        joint_state = features[joint_key]
        action = features["action"]
        tactile = features.get("observation.tactile")
        tcp = features.get("observation.state.tcp")
        joint_torque = features.get("observation.state.joint_torque")

        def image_size(key: str) -> tuple[int, int]:
            feature = features.get(key)
            if feature is None:
                return (0, 0)
            shape = _to_int_list(feature.get("shape") or [])
            if len(shape) >= 2:
                return (int(shape[0]), int(shape[1]))
            return (0, 0)

        action_names = action.get("names") or []
        return cls(
            fps=float(info["fps"]),
            action_dim=_to_int_list(action["shape"])[0],
            state_dim=_to_int_list(joint_state["shape"])[0],
            joint_torque_dim=(
                _to_int_list(joint_torque["shape"])[0] if joint_torque is not None else 0
            ),
            tcp_dim=_to_int_list(tcp["shape"])[0] if tcp is not None else 0,
            tactile_dim=_to_int_list(tactile["shape"])[0] if tactile is not None else 0,
            action_names=action_names,
            camera_size=image_size("observation.images.head_left"),
            tactile_raw_size=image_size("observation.images.tactile_raw"),
            tactile_deform_size=image_size("observation.images.tactile_deform"),
        )

    @classmethod
    def load(cls, season_root: Path) -> "DatasetMetadata":
        """Load metadata from a season's ``meta`` folder.

        Args:
            season_root: The ``lerobot3.0/`` directory of a season.

        Returns:
            A populated :class:`DatasetMetadata` instance.
        """
        meta_dir = Path(season_root) / "meta"
        info = _read_json(meta_dir / "info.json")
        modality = _read_json(meta_dir / "modality.json")
        return cls.from_info(info, modality)


@dataclass
class FeatureStats:
    """Per-feature statistics used for normalization.

    Stored in ``meta/stats.json`` and used to build the action/state normalizers.
    """

    min: list[float] = field(default_factory=list)
    max: list[float] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    q01: list[float] = field(default_factory=list)
    q99: list[float] = field(default_factory=list)
    count: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FeatureStats":
        """Build stats from the parsed ``meta/stats.json`` section for one feature."""
        return cls(
            min=list(data.get("min") or []),
            max=list(data.get("max") or []),
            mean=list(data.get("mean") or []),
            std=list(data.get("std") or []),
            q01=list(data.get("q01") or []),
            q99=list(data.get("q99") or []),
            count=list(data.get("count") or []),
        )


class DatasetStats:
    """Collection of :class:`FeatureStats` keyed by feature name.

    Attributes:
        by_feature: Mapping from feature key to its :class:`FeatureStats`.
    """

    def __init__(self, by_feature: Dict[str, FeatureStats]) -> None:
        """Initialise from a feature->stats mapping.

        Args:
            by_feature: Mapping of feature key to :class:`FeatureStats`.
        """
        self.by_feature = by_feature

    @classmethod
    def load(cls, season_root: Path) -> "DatasetStats":
        """Load stats from a season's ``meta/stats.json``.

        Args:
            season_root: The ``lerobot3.0/`` directory of a season.

        Returns:
            A :class:`DatasetStats` instance.
        """
        stats = _read_json(Path(season_root) / "meta" / "stats.json")
        return cls({key: FeatureStats.from_json(value) for key, value in stats.items()})

    @staticmethod
    def _pad(vector: list[float], size: int) -> list[float]:
        """Extend ``vector`` to ``size`` with trailing zeros (or trim)."""
        vector = list(vector)
        return (vector + [0.0] * size)[:size]

    def action(self, dim: int) -> FeatureStats:
        """Return action stats, padded/trimmed to ``dim``."""
        stats = self.by_feature["action"]
        stats.std = self._pad(stats.std, dim)
        stats.mean = self._pad(stats.mean, dim)
        return stats

    def state(self, dim: int) -> FeatureStats:
        """Return per-joint state stats, padded/trimmed to ``dim``."""
        stats = self.by_feature["observation.state"]
        stats.std = self._pad(stats.std, dim)
        stats.mean = self._pad(stats.mean, dim)
        return stats

    def anything(self, key: str) -> FeatureStats:
        """Return raw stats for an arbitrary feature key."""
        return self.by_feature[key]
