"""Whether the GUI should run Real-ESRGAN on each crop.

Off by default: the landmark head (and the multiclass and germination models) were built on plain crops, and
super-resolved crops raised the landmark ensemble's validation error from 20.6 to 34.3 deg. Set DLHOOK_SUPERRES=1 to reproduce older exports that used it (needs CUDA)."""
from __future__ import annotations

import os

ENV_VAR = "DLHOOK_SUPERRES"


def superres_requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return str(environ.get(ENV_VAR, "")).strip().lower() in ("1", "true", "yes", "on")
