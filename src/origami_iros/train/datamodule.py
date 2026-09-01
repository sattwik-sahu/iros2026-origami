"""Lightning data module wrapping the LeRobot season datasets.

Converts the raw LeRobot season data into two PyTorch ``DataLoader``s (train and
validation) that yield ``(Observation, action, action_is_pad)`` batches suitable
for the VLTA policy.
"""

from __future__ import annotations

import lightning as L
from typing import override
from torch.utils.data import DataLoader

from origami_iros.data.collate import vlta_collate_fn
from origami_iros.data.dataset import build_train_val_datasets
from origami_iros.data.preprocessing import ObservationPreprocessor
from origami_iros.train.config import DataConfig


class VLTA_pl_datamodule(L.LightningDataModule):
    """Lightning data module for the origami LeRobot season datasets.

    Attributes:
        config: The data configuration (paths, batch size, workers, chunking).
        fps: Frames per second of the dataset (from metadata, resolved upstream).
        preprocessor: Preprocessor applied to each raw sample.
        _train_loader: Lazily constructed training data loader.
        _val_loader: Lazily constructed validation data loader.
    """

    def __init__(
        self,
        config: DataConfig,
        fps: float,
        preprocessor: ObservationPreprocessor | None = None,
    ) -> None:
        """Initialise the data module.

        Args:
            config: The data configuration.
            fps: Frames per second of the dataset (from metadata).
            preprocessor: Optional preprocessor applied to each sample.
        """
        super().__init__()
        self.config = config
        self.fps = fps
        self.preprocessor = preprocessor

    @override
    def setup(self, stage: str | None = None) -> None:
        """Build the train and validation datasets and data loaders.

        Args:
            stage: Lightning stage (``"fit"``, ``"validate"``, ``"test"``, ...).
        """
        cfg = self.config
        # The policy predicts a chunk of future actions, so each sample carries a
        # window of ``chunk_size`` action timesteps spaced ``1/fps`` apart.
        delta_timestamps = {
            "action": [i / self.fps for i in range(cfg.chunk_size)]
        }

        train_ds, val_ds = build_train_val_datasets(
            cfg.data_root,
            delta_timestamps,
            preprocessor=self.preprocessor,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            dataset_subdir=cfg.dataset_subdir,
            video_backend=cfg.video_backend,
        )

        common = dict(
            batch_size=cfg.batch_size,
            collate_fn=vlta_collate_fn,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )
        self._train_loader = DataLoader(
            train_ds, shuffle=True, drop_last=True, **common
        )
        self._val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            batch_size=cfg.batch_size,
            collate_fn=vlta_collate_fn,
            num_workers=max(1, cfg.num_workers // 2),
            persistent_workers=cfg.num_workers > 0,
        )

    @override
    def train_dataloader(self) -> DataLoader:
        """Return the training data loader.

        Returns:
            The training data loader.
        """
        return self._train_loader

    @override
    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader.

        Returns:
            The validation data loader.
        """
        return self._val_loader
