# origami_iros/modules/data/lerobot_seasons.py
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import FrameTimestampError


class ResilientDataset(Dataset):
    def __init__(self, dataset: Dataset, max_retries: int = 5) -> None:
        self._dataset = dataset
        self._max_retries = max_retries

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        for _ in range(self._max_retries):
            try:
                return self._dataset[idx]
            except FrameTimestampError:
                idx = random.randrange(len(self._dataset))
        raise RuntimeError(f"exceeded {self._max_retries} retries fetching a valid sample")


def discover_seasons(root: str | Path, dataset_subdir: str = "lerobot3.0") -> list[Path]:
    root = Path(root)
    return sorted(p / dataset_subdir for p in root.iterdir() if p.is_dir() and (p / dataset_subdir).exists())


def build_season_dataset(
    season_root: Path,
    delta_timestamps: dict,
    video_backend: str = "pyav",
    tolerance_s: float = 1e-2,
) -> LeRobotDataset:
    try:
        ds = LeRobotDataset(
            repo_id=season_root.parent.name,
            root=season_root,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
            tolerance_s=tolerance_s,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build dataset for season {season_root}: {e}")
    return ds


def build_train_val_datasets(
    root: str | Path,
    delta_timestamps: dict,
    val_fraction: float = 0.1,
    seed: int = 0,
    dataset_subdir: str = "lerobot3.0",
    fps: float = 30.0,
) -> tuple[ConcatDataset, ConcatDataset]:
    season_roots = discover_seasons(root, dataset_subdir)
    if not season_roots:
        raise RuntimeError(f"no seasons found under {root}")

    tolerance_s = 0.5 / fps

    rng = random.Random(seed)
    shuffled = season_roots[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_roots = sorted(shuffled[:n_val])
    train_roots = sorted(shuffled[n_val:])

    train_ds = ConcatDataset(
        [ResilientDataset(build_season_dataset(r, delta_timestamps, tolerance_s=tolerance_s)) for r in train_roots]
    )
    val_ds = ConcatDataset(
        [ResilientDataset(build_season_dataset(r, delta_timestamps, tolerance_s=tolerance_s)) for r in val_roots]
    )

    print(f"→ {len(train_roots)} seasons train ({len(train_ds)} frames), {len(val_roots)} seasons val ({len(val_ds)} frames)")
    return train_ds, val_ds