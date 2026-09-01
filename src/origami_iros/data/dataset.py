"""Dataset construction: season discovery, Lerobot loading, preprocessing.

The dataset is a set of *seasons* (closed-loop collection sessions) stored under a
single root folder, e.g. ``dataset/season_POC22032_.../lerobot3.0``. Each season
is a self-describing Lerobot v3.0 dataset.

Dataset parameters (fps, action dimension, tactile dimension, ...) are **read from
the season metadata** (``meta/info.json``) instead of being hardcoded in the
training configuration. This guarantees the data pipeline stays in sync with the
recorded data. Preprocessing (whitening of scalar features from ``meta/stats.json``)
is applied on top of each Lerobot sample.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError
from typing import override
from torch.utils.data import ConcatDataset, Dataset

from origami_iros.data.metadata import DatasetMetadata, DatasetStats
from origami_iros.data.preprocessing import ObservationPreprocessor


class PreprocessedSeasonDataset(Dataset):
    """Thin wrapper that applies a preprocessor to each raw Lerobot sample.

    Attributes:
        _dataset: The wrapped Lerobot dataset.
        _preprocessor: Optional preprocessor applied to each sample.
    """

    def __init__(
        self,
        dataset: Dataset,
        preprocessor: ObservationPreprocessor | None = None,
        max_retries: int = 5,
    ) -> None:
        """Initialise the wrapper.

        Args:
            dataset: The underlying (Lerobot) dataset.
            preprocessor: Optional preprocessor applied to each sample.
            max_retries: Number of retries if a frame timestamp error occurs.
        """
        self._dataset = dataset
        self._preprocessor = preprocessor
        self._max_retries = max_retries

    @override
    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(self._dataset)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Fetch and preprocess a sample, retrying on frame-timestamp errors.

        Args:
            idx: Sample index.

        Returns:
            The (preprocessed) sample dict.
        """
        for _ in range(self._max_retries):
            try:
                sample = self._dataset[idx]
            except FrameTimestampError:
                idx = random.randrange(len(self._dataset))
                continue
            if self._preprocessor is not None:
                sample = self._preprocessor(sample)
            return sample
        raise RuntimeError(f"exceeded {self._max_retries} retries fetching a valid sample")


def discover_seasons(root: str | Path, dataset_subdir: str = "lerobot3.0") -> list[Path]:
    """List the Lerobot season roots under a dataset root folder.

    Args:
        root: Parent folder containing the season directories.
        dataset_subdir: Name of the Lerobot sub-folder inside each season
            (``lerobot3.0`` by default).

    Returns:
        Sorted list of paths to each season's Lerobot root.
    """
    root = Path(root)
    return sorted(
        p / dataset_subdir
        for p in root.iterdir()
        if p.is_dir() and (p / dataset_subdir).exists()
    )


def estimate_fps(season_root: Path) -> float:
    """Return the fps recorded for a season from its ``info.json``.

    Args:
        season_root: The season's Lerobot root directory.

    Returns:
        The dataset frames-per-second.
    """
    return DatasetMetadata.load(season_root).fps


def build_season_dataset(
    season_root: Path,
    delta_timestamps: dict,
    preprocessor: ObservationPreprocessor | None = None,
    video_backend: str = "pyav",
    tolerance_s: float = 1e-2,
) -> PreprocessedSeasonDataset:
    """Build a single preprocessed season dataset.

    Args:
        season_root: The season's Lerobot root directory.
        delta_timestamps: Lerobot delta-timestamps mapping (e.g. action offsets).
        preprocessor: Optional preprocessor applied to each sample.
        video_backend: Video decoding backend passed to Lerobot.
        tolerance_s: Sync tolerance in seconds passed to Lerobot.

    Returns:
        A :class:`PreprocessedSeasonDataset` wrapping the Lerobot dataset.

    Raises:
        RuntimeError: If the Lerobot dataset could not be built.
    """
    try:
        ds = LeRobotDataset(
            repo_id=season_root.parent.name,
            root=season_root,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build dataset for season {season_root}: {exc}") from exc
    return PreprocessedSeasonDataset(ds, preprocessor=preprocessor)


def build_train_val_datasets(
    root: str | Path,
    delta_timestamps: dict,
    preprocessor: ObservationPreprocessor | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
    dataset_subdir: str = "lerobot3.0",
    video_backend: str = "pyav",
) -> tuple[ConcatDataset, ConcatDataset]:
    """Split seasons into train/validation and build concatenated datasets.

    The fps, used to derive the Lerobot sync tolerance, is read from the first
    season's metadata rather than being hardcoded.

    Args:
        root: Parent folder containing the season directories.
        delta_timestamps: Lerobot delta-timestamps mapping.
        preprocessor: Optional preprocessor applied to each sample.
        val_fraction: Fraction of seasons held out for validation.
        seed: Random seed used to shuffle the season split.
        dataset_subdir: Lerobot sub-folder name inside each season.
        video_backend: Video decoding backend passed to Lerobot.

    Returns:
        A tuple ``(train_dataset, val_dataset)`` of concatenated season datasets.

    Raises:
        RuntimeError: If no season is found under ``root``.
    """
    season_roots = discover_seasons(root, dataset_subdir)
    if not season_roots:
        raise RuntimeError(f"no seasons found under {root}")

    fps = DatasetMetadata.load(season_roots[0]).fps
    tolerance_s = 0.5 / fps

    rng = random.Random(seed)
    shuffled = season_roots[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_roots = sorted(shuffled[:n_val])
    train_roots = sorted(shuffled[n_val:])

    train_ds = ConcatDataset(
        [
            build_season_dataset(
                r, delta_timestamps, preprocessor=preprocessor, video_backend=video_backend,
                tolerance_s=tolerance_s,
            )
            for r in train_roots
        ]
    )
    val_ds = ConcatDataset(
        [
            build_season_dataset(
                r, delta_timestamps, preprocessor=preprocessor, video_backend=video_backend,
                tolerance_s=tolerance_s,
            )
            for r in val_roots
        ]
    )

    print(
        f"→ {len(train_roots)} seasons train ({len(train_ds)} frames), "
        f"{len(val_roots)} seasons val ({len(val_ds)} frames) @ {fps} fps"
    )
    return train_ds, val_ds


def build_preprocessor_from_seasons(
    season_roots: list[Path],
    action_dim: int | None = None,
    state_dim: int | None = None,
    normalize_actions: bool = True,
    normalize_state: bool = True,
    normalize_tactile: bool = True,
    normalize_torque: bool = True,
) -> ObservationPreprocessor:
    """Build the sample preprocessor from a season's statistics.

    Args:
        season_roots: Season Lerobot roots (the statistics are read from the first).
        action_dim: Action dimension used to pad/trim the action statistics.
        state_dim: State dimension used to pad/trim the state statistics.
        normalize_actions: Whitening for actions.
        normalize_state: Whitening for joint state.
        normalize_tactile: Whitening for tactile state.
        normalize_torque: Whitening for joint torque.

    Returns:
        A configured :class:`ObservationPreprocessor`.
    """
    if not season_roots:
        raise ValueError("at least one season root is required")
    stats = DatasetStats.load(season_roots[0])
    return ObservationPreprocessor.from_dataset_stats(
        stats,
        normalize_actions=normalize_actions,
        normalize_state=normalize_state,
        normalize_tactile=normalize_tactile,
        normalize_torque=normalize_torque,
        action_dim=action_dim,
        state_dim=state_dim,
    )
