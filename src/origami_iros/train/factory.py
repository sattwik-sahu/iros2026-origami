"""Build runtime configuration objects from the dataset metadata.

The training pipeline deliberately derives *data facts* (fps, action dimension,
proprioceptive dimensions, camera / tactile image sizes) from each LeRobot
season's ``meta/info.json`` rather than hardcoding them in the Hydra config.
This module is the single place that resolves those facts and turns them into the
concrete values used to build the data loaders, the model, the preprocessors, and
the hub normalizer.

Every resolved value can still be overridden by the user through the config (the
``*_override`` style fields), but by default it follows the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from origami_iros.data.dataset import discover_seasons
from origami_iros.data.metadata import DatasetMetadata, DatasetStats
from origami_iros.data.preprocessing import ObservationPreprocessor
from origami_iros.train.config import DataConfig, ModelConfig


@dataclass
class ResolvedDataFacts:
    """Concrete, metadata-derived data facts used throughout the pipeline.

    Attributes:
        fps: Frames per second of the dataset.
        action_dim: Dimension of a single action vector.
        torque_dim: Dimension of the joint-torque stream.
        joint_state_dim: Dimension of the joint-state stream.
        proprio_tactile_dim: Dimension of the proprioceptive tactile signal.
        camera_image_size: ``(h, w)`` of each camera view.
        tactile_image_size: ``(h, w)`` of the primary (raw) tactile image.
    """

    fps: float
    action_dim: int
    torque_dim: int
    joint_state_dim: int
    proprio_tactile_dim: int
    camera_image_size: tuple[int, int]
    tactile_image_size: tuple[int, int]

    @classmethod
    def from_metadata(cls, metadata: DatasetMetadata) -> "ResolvedDataFacts":
        """Build facts from a single season's resolved metadata.

        Args:
            metadata: The season metadata.

        Returns:
            A :class:`ResolvedDataFacts` instance.
        """
        return cls(
            fps=metadata.fps,
            action_dim=metadata.action_dim,
            torque_dim=(
                metadata.joint_torque_dim
                if metadata.joint_torque_dim > 0
                else metadata.state_dim
            ),
            joint_state_dim=metadata.state_dim,
            proprio_tactile_dim=metadata.tactile_dim,
            camera_image_size=metadata.camera_size or (480, 480),
            # Use the raw tactile stream as the primary signal (the deform
            # stream is all zeros in the recorded data).
            tactile_image_size=metadata.tactile_raw_size or (480, 1200),
        )


def resolve_data_facts(
    data_root: str | Path,
    dataset_subdir: str,
    data: DataConfig,
) -> tuple[ResolvedDataFacts, list[Path]]:
    """Resolve concrete data facts and season roots for a dataset root.

    Args:
        data_root: Parent folder containing the season directories.
        dataset_subdir: Name of the Lerobot sub-folder inside each season.
        data: The user configuration (used for optional overrides).

    Returns:
        A ``(facts, season_roots)`` tuple. ``facts`` are the concrete,
        metadata-derived values; ``season_roots`` are the discovered season
        Lerobot roots.

    Raises:
        RuntimeError: If no season is found under the root.
    """
    season_roots = discover_seasons(data_root, dataset_subdir)
    if not season_roots:
        raise RuntimeError(f"no seasons found under {data_root}")

    metadata = DatasetMetadata.load(season_roots[0])
    facts = ResolvedDataFacts.from_metadata(metadata)

    if data.fps_override is not None:
        facts.fps = data.fps_override
    return facts, season_roots


def resolve_model_config(model: ModelConfig, facts: ResolvedDataFacts) -> ModelConfig:
    """Fill the data-derived fields of a model config from resolved facts.

    Any field left as ``None`` in the user config is taken from ``facts``; fields
    the user set explicitly are preserved.

    Args:
        model: The user model configuration.
        facts: The resolved data facts.

    Returns:
        A fully-populated :class:`ModelConfig`.
    """
    return ModelConfig(
        vit_model_name=model.vit_model_name,
        image_size=model.image_size or facts.camera_image_size,
        vit_dim=model.vit_dim,
        freeze_vit=model.freeze_vit,
        tactile_image_size=model.tactile_image_size or facts.tactile_image_size,
        tactile_patch_size=model.tactile_patch_size,
        n_hands=model.n_hands,
        n_fingers=model.n_fingers,
        torque_dim=(model.torque_dim or facts.torque_dim),
        joint_state_dim=(model.joint_state_dim or facts.joint_state_dim),
        proprio_tactile_dim=(model.proprio_tactile_dim or facts.proprio_tactile_dim),
        action_dim=(model.action_dim or facts.action_dim),
        tactile_dim=model.tactile_dim,
        hidden_dim=model.hidden_dim,
        action_hidden_dim=model.action_hidden_dim,
        action_num_layers=model.action_num_layers,
        action_num_heads=model.action_num_heads,
        num_inference_steps=model.num_inference_steps,
    )


def build_preprocessor(
    season_roots: list[Path],
    facts: ResolvedDataFacts,
    data: DataConfig,
) -> ObservationPreprocessor:
    """Build the sample preprocessor (whitening) from dataset statistics.

    Aggregates stats across *all* seasons (pooled mean/std) rather than using
    only the first season, which would bias whitening because seasons have
    slightly different means (e.g. action mean 1.01 vs 0.99) and tiny-std dims.

    Args:
        season_roots: The season Lerobot roots.
        facts: The resolved data facts (used to pad the statistics).
        data: The user data configuration (preprocessing toggles).

    Returns:
        A configured :class:`ObservationPreprocessor`.
    """
    # Aggregate stats from all seasons (pooled)
    from origami_iros.data.metadata import DatasetStats as _DS
    import numpy as np

    all_stats = [_DS.load(r) for r in season_roots]
    # Use first as base, then average across seasons weighted by count
    base = all_stats[0]
    # For each feature, pooled mean/std via weighted average (approx)
    # We do simple average of means/stds weighted by count, which is sufficient
    # for whitening robustness; exact pooled variance would need sum of squares.
    merged = {}
    for key in base.by_feature:
        # Collect per-season FeatureStats for this key
        feats = [s.by_feature[key] for s in all_stats if key in s.by_feature]
        if not feats:
            continue
        counts = np.array([f.count[0] if f.count else 1 for f in feats], dtype=float)
        w = counts / counts.sum()
        # Weighted average for mean/std/q01/q99
        def wavg(attr):
            arrs = [np.asarray(getattr(f, attr), dtype=float) for f in feats]
            # Pad to same length (action dim)
            max_len = max(len(a) for a in arrs)
            padded = [np.pad(a, (0, max_len - len(a))) for a in arrs]
            stacked = np.stack(padded, axis=0)
            return np.average(stacked, axis=0, weights=w).tolist()

        merged[key] = base.by_feature[key].__class__(
            min=wavg("min"),
            max=wavg("max"),
            mean=wavg("mean"),
            std=wavg("std"),
            q01=wavg("q01"),
            q99=wavg("q99"),
            count=[int(counts.sum())],
        )
    stats = _DS(merged)
    return ObservationPreprocessor.from_dataset_stats(
        stats,
        normalize_actions=data.normalize_actions,
        normalize_state=data.normalize_state,
        normalize_tactile=data.normalize_tactile,
        normalize_torque=data.normalize_torque,
        action_dim=facts.action_dim,
        state_dim=facts.joint_state_dim,
    )
