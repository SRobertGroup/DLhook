"""Re-export shim: moved to models/multiclass_inference.py, the shared home
for inference-time code that both multi/ and the GUI depend on (see
CLAUDE.md's multi/ boundary note: multi/ depends on models/, never the
reverse). Kept here, unchanged in behaviour, so existing imports
(multi/evaluate_multiclass.py, multi/evaluate_production_domain.py,
multi/sweep_operating_points.py, and `from src.multiclass_inference import
...` at those call sites) keep working. `zscore_normalize` is re-exported
too since it was a module-level name here before the move (some tests check
identity against it directly)."""
from __future__ import annotations

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

from models.multiclass_inference import (  # noqa: E402,F401
    CPU_BATCH_SIZE,
    MulticlassInference,
    zscore_normalize,
)
