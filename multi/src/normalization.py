"""Re-export shim: moved to models/normalization.py, the shared home for
inference-time code that both multi/ and the GUI depend on (see CLAUDE.md's
multi/ boundary note: multi/ depends on models/, never the reverse). Kept
here, unchanged in behaviour, so existing imports of multi.src.normalization
(and `from .normalization import ...` within multi/src/) keep working."""
from __future__ import annotations

from .config import ensure_repo_root_importable

ensure_repo_root_importable()

from models.normalization import DEFAULT_EPS, zscore_normalize  # noqa: E402,F401
