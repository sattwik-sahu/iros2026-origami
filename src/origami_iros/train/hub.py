"""Export the trained policy to the Hugging Face Hub.

At the end of training the user may optionally push the trained model so others
can load it and run inference directly. Uploaded artefacts mirror what a client
needs to reconstruct inference:

* ``model.safetensors`` - the policy state dict.
* ``config.json`` - model architecture plus the resolved data facts.
* ``normalizer.json`` - action mean/std (from the dataset stats) used to
  unnormalize sampled actions into real robot units.
* ``README.md`` - human-readable usage instructions.

The normalizer is essential: the policy is trained on whitened actions, so the
client must unnormalize the predicted action before sending it to the robot.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from huggingface_hub import HfApi
from typing import override

from origami_iros.data.metadata import DatasetStats, FeatureStats
from origami_iros.models.policy.vlta_policy import VLTAPolicy
from origami_iros.train.factory import ResolvedDataFacts


def _serializable_stats(feature_stats: FeatureStats) -> dict[str, Any]:
    """Convert :class:`FeatureStats` into a JSON-serialisable dict."""
    return {
        "min": feature_stats.min,
        "max": feature_stats.max,
        "mean": feature_stats.mean,
        "std": feature_stats.std,
    }


def build_normalizer(season_root: Path, action_dim: int | None = None) -> dict[str, Any]:
    """Build the JSON-serialisable normalizer from a season's statistics.

    Args:
        season_root: The season Lerobot root directory.
        action_dim: Action dimension used to pad/trim the action statistics.

    Returns:
        A dict with ``action`` and ``observation.state`` statistics.
    """
    stats = DatasetStats.load(season_root)
    normalizer: dict[str, Any] = {}
    if "action" in stats.by_feature:
        action = stats.action(action_dim or 0)
        normalizer["action"] = _serializable_stats(action)
    if "observation.state" in stats.by_feature:
        normalizer["observation.state"] = _serializable_stats(
            stats.anything("observation.state")
        )
    return {"type": "mean_std", **normalizer}


def _config_to_json(model_config, facts: ResolvedDataFacts) -> dict[str, Any]:
    """Build the exported ``config.json`` from the model config and data facts."""
    from dataclasses import fields

    def _clean(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value

    cfg = {
        field.name: _clean(getattr(model_config, field.name))
        for field in fields(model_config)
    }
    facts_dict = asdict(facts)
    facts_dict = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in facts_dict.items()
    }
    return {"model": cfg, "data_facts": facts_dict}


def push_model_to_hub(
    repo_id: str,
    model: VLTAPolicy,
    model_config,
    facts: ResolvedDataFacts,
    season_root: Path,
    private: bool = True,
    readme: str | None = None,
) -> None:
    """Write artefacts and upload them to the Hugging Face Hub.

    Args:
        repo_id: Destination repo id (``owner/repo``).
        model: The trained policy.
        model_config: Model configuration used to build the policy.
        facts: Resolved data facts describing the training data.
        season_root: Season Lerobot root used to source the normalizer stats.
        private: Whether the created repo should be private.
        readme: Optional README markdown to store as ``README.md`` on the hub.

    Raises:
        RuntimeError: If the upload fails.
    """
    api = HfApi()
    # ``private`` only matters at repo creation; an existing public repo is kept
    # as-is rather than silently changed.
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not create/create_or_get hub repo {repo_id}: {exc}") from exc

    import tempfile

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        save_file(model.state_dict(), tmpdir / "model.safetensors")
        (tmpdir / "config.json").write_text(
            json.dumps(_config_to_json(model_config, facts), indent=2), encoding="utf-8"
        )
        (tmpdir / "normalizer.json").write_text(
            json.dumps(build_normalizer(season_root, facts.action_dim), indent=2),
            encoding="utf-8",
        )
        if readme is not None:
            (tmpdir / "README.md").write_text(readme, encoding="utf-8")

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(tmpdir),
            commit_message="Upload VLTA flow-matching policy",
        )


class HubPushCallback(Callback):
    """Push the trained policy to the Hugging Face Hub at the end of training.

    Attributes:
        repo_id: Destination repo id (``owner/repo``).
        private: Whether to create the repo private.
        season_root: Season Lerobot root used to source the normalizer.
        facts: Resolved data facts describing the training data.
        readme: Optional README markdown stored on the hub.
    """

    def __init__(
        self,
        repo_id: str,
        private: bool,
        season_root: Path,
        facts: ResolvedDataFacts,
        readme: str | None = None,
    ) -> None:
        """Initialise the callback.

        Args:
            repo_id: Destination repo id (``owner/repo``).
            private: Whether to create the repo private.
            season_root: Season Lerobot root used to source the normalizer.
            facts: Resolved data facts describing the training data.
            readme: Optional README markdown stored on the hub.
        """
        super().__init__()
        self.repo_id = repo_id
        self.private = private
        self.season_root = season_root
        self.facts = facts
        self.readme = readme
        self._pushed = False

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Push once the model has finished training.

        Args:
            trainer: The Lightning trainer.
            pl_module: The Lightning module wrapping the policy.
        """
        if self._pushed:
            return
        model = getattr(pl_module, "model", pl_module)
        model_config = getattr(pl_module, "model_config", None)
        push_model_to_hub(
            repo_id=self.repo_id,
            model=model,
            model_config=model_config,
            facts=self.facts,
            season_root=self.season_root,
            private=self.private,
            readme=self.readme,
        )
        self._pushed = True
