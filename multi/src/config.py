"""Config loading for the multi/ multiclass training pipeline.

Paths in the YAML config are declared relative to a "data root", resolved in
this order: the explicit `data_root` argument to `load_config`, else the
`DLHOOK_DATA_ROOT` environment variable, else the repository root (this file
lives at <repo>/multi/src/config.py, three levels down).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# multi/src/config.py -> src -> multi -> <repo root>
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_repo_root_importable() -> None:
    """Make the repo root importable so `models.*` / `utils.*` can be reached
    from this package. multi/merge_masks.py et al. put `multi/` (not the repo
    root) on sys.path so that `import src...` resolves to `multi/src`; the
    repo root is added here, additively, only by the modules that actually
    need first-party GUI packages (mask_merge.py for models.UNetInference,
    model.py for models.unet)."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def load_config(path: str | Path, data_root: str | Path | None = None) -> dict:
    """Load a training_config.yaml and attach the data root it should be
    resolved against (see module docstring). Also validates that
    `data.num_classes` and `model.num_classes` agree -- these are two
    separate keys in the YAML with no other cross-check."""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if data_root is None:
        data_root = os.environ.get("DLHOOK_DATA_ROOT")
    config["_data_root"] = str(data_root) if data_root is not None else None

    data_num_classes = config.get("data", {}).get("num_classes")
    model_num_classes = config.get("model", {}).get("num_classes")
    if data_num_classes != model_num_classes:
        raise ValueError(
            "Config validation failed: data.num_classes "
            f"({data_num_classes!r}) != model.num_classes ({model_num_classes!r}). "
            "These must be kept in sync in the YAML."
        )

    return config


def resolved_path(config: dict, key: str, section: str = "paths") -> Path:
    """Resolve `config[section][key]` against the config's data root.
    `section` defaults to "paths" (the training-run paths); pass
    section="evaluation" for the validation-only keys (annotations_*, and
    the raw dir the 441 human-annotated crops live under), which are kept
    in a separate config section since they are never used by training."""
    raw = config[section][key]
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return raw_path

    data_root = config.get("_data_root")
    if data_root:
        root = Path(data_root)
    else:
        root = REPO_ROOT
    return root / raw_path
