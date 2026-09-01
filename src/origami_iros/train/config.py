"""Structured configuration dataclasses for training runs.

These dataclasses define the typed schema behind the Hydra configuration files
under ``configs/``. Hydra instantiates them from YAML at runtime so every
hyperparameter is declared in one place, is validated, and is serialised to
WandB and checkpoints automatically.

Data facts that are intrinsic to the recorded dataset (fps, action dimension,
tactile dimension, image sizes) are **derived from the dataset metadata at
runtime** rather than hardcoded here, so the config never drifts from the data.
The fields that would otherwise encode those facts (e.g. ``DataConfig.fps``,
``ModelConfig.action_dim``) are therefore optional overrides that, when left
unset (``None``), fall back to the values read from ``meta/info.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Configuration for the LeRobot season dataset and loading.

    Tuned for RTX A4500 (20GB VRAM, 32 threads): batch 48 uses ~5.9GB training
    peak, leaving headroom for optimizer states/activations; 12 workers
    saturate CPU decode without thrashing.
    """

    data_root: str = "dataset"
    dataset_subdir: str = "lerobot3.0"
    batch_size: int = 64
    num_workers: int = 8
    val_fraction: float = 0.2
    val_batches: int = 20
    chunk_size: int = 13
    video_backend: str = "pyav"
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    seed: int = 0

    # Optional overrides for data facts derived from metadata. When set to None
    # the values are read from the first season's `meta/info.json`.
    fps_override: Optional[float] = None

    # Whether/when to normalize the scalar modalities during preprocessing.
    normalize_actions: bool = True
    normalize_state: bool = True
    normalize_tactile: bool = True
    normalize_torque: bool = True


@dataclass
class ModelConfig:
    """Configuration for the image encoders, state encoders and action head.

    Dimension fields that describe the *recorded data* (``action_dim``,
    ``torque_dim``, ``joint_state_dim``, ``proprio_tactile_dim``, image sizes)
    default to ``None`` and are filled in from the dataset metadata when the
    policy is built. Fields that describe pure *architecture* choices are kept
    as explicit values here.
    """

    vit_model_name: str = "facebook/dinov2-small"
    image_size: Optional[tuple[int, int]] = None
    vit_dim: int = 384
    freeze_vit: bool = True

    tactile_image_size: Optional[tuple[int, int]] = None
    tactile_patch_size: int = 16
    n_hands: int = 2
    n_fingers: int = 5

    # Data-derived dimensions (default None -> filled from metadata).
    torque_dim: Optional[int] = None
    joint_state_dim: Optional[int] = None
    proprio_tactile_dim: Optional[int] = None
    action_dim: Optional[int] = None
    tactile_dim: int = 192

    hidden_dim: int = 512
    action_hidden_dim: int = 512
    action_num_layers: int = 6
    action_num_heads: int = 8
    num_inference_steps: int = 10


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer and learning-rate schedule."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Learning-rate schedule — warmup 500/20000=2.5% (was 1000, loss flat for 1k steps)
    # warmup_start_lr 3e-6 ensures warmup ramps 3e-6 -> lr (1e-4), not 0 -> lr
    scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 500
    warmup_start_lr: float = 3e-6
    total_steps: int = 20000
    lr_min: float = 3e-6


@dataclass
class CallbacksConfig:
    """Configuration for callbacks, logging and checkpointing cadence."""

    log_every_n_steps: int = 10
    val_every_n_steps: int = 1000
    checkpoint_every_n_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    num_sample_replay: int = 4


@dataclass
class WandBConfig:
    """Configuration for the Weights & Biases logger."""

    project: str = "iros2026-origami"
    entity: Optional[str] = "building-text"
    name: Optional[str] = None
    tags: tuple[str, ...] = ()
    save_code: bool = True


@dataclass
class HubConfig:
    """Configuration for optionally pushing the model to the Hugging Face Hub.

    Note:
        When ``push`` is ``True`` the model is uploaded at the end of training to
        the repo ``repo_id`` (must be of the form ``owner/repo``). Authenticate
        beforehand with ``huggingface-cli login``.
    """

    push: bool = True
    repo_id: str = "sattwik21/sharpa-north-origami"
    private: bool = False


@dataclass
class TrainConfig:
    """Top-level configuration grouping all sub-configurations."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    seed: Optional[int] = 0
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    max_steps: int = 20000
    accumulate_grad_batches: int = 2
    run_name: Optional[str] = None
    deterministic: bool = False

    @classmethod
    def from_dictconfig(cls, cfg) -> "TrainConfig":
        """Build a ``TrainConfig`` from a composed Hydra ``DictConfig``.

        Args:
            cfg: The composed ``OmegaConf.DictConfig`` produced by ``@hydra.main``.

        Returns:
            A fully-populated, typed :class:`TrainConfig`.
        """
        from hydra.utils import instantiate

        return instantiate(cfg, _recursive_=True)


def register_config_store() -> None:
    """Register the structured config schema with Hydra's ``ConfigStore``.

    This lets ``@hydra.main`` validate and build the composed ``config.yaml``
    against the typed dataclasses above, and lets ``hydra.utils.instantiate``
    reconstruct a fully-populated :class:`TrainConfig` at runtime.
    """
    cs = ConfigStore.instance()
    cs.store(name="config", node=TrainConfig)
    cs.store(group="data", name="default", node=DataConfig)
    cs.store(group="model", name="default", node=ModelConfig)
    cs.store(group="optimizer", name="default", node=OptimizerConfig)
    cs.store(group="callbacks", name="default", node=CallbacksConfig)
    cs.store(group="wandb", name="default", node=WandBConfig)
    cs.store(group="hub", name="default", node=HubConfig)

