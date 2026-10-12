"""The torch device every model runs on, chosen once per process.

`torch.cuda.is_available()` alone is not trusted: it can report True for a GPU the installed
torch build has no kernels for (the sm_120 / Blackwell mismatch recorded in update_plan.md),
which then fails late, at the first real forward pass. pick_device() runs a tiny CUDA op first
and falls back to the CPU, with one warning, when that op fails.

DLHOOK_DEVICE=cpu|cuda overrides the choice (cuda is still probed, and still falls back).
"""
import logging
import os

import torch

log = logging.getLogger(__name__)

DEVICE_ENV = "DLHOOK_DEVICE"
VALID_DEVICES = ("cpu", "cuda")

_DEVICE_CACHE = []


def _cuda_works():
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.ones(2, device="cuda") * 2
        return bool(probe.cpu().sum().item() == 4.0)
    except Exception as exc:                     # any CUDA failure means "use the CPU"
        log.warning(f"CUDA is reported available but a test operation failed ({exc}); using the CPU")
        return False


def pick_device(environ=None):
    """The process-wide torch.device for inference (cached after the first call)."""
    if _DEVICE_CACHE:
        return _DEVICE_CACHE[0]
    env = os.environ if environ is None else environ
    requested = (env.get(DEVICE_ENV) or "").strip().lower()
    if requested and requested not in VALID_DEVICES:
        raise ValueError(f"{DEVICE_ENV}={requested!r}: expected one of {VALID_DEVICES}")
    if requested == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if _cuda_works() else "cpu")
        if requested == "cuda" and device.type == "cpu":
            log.warning(f"{DEVICE_ENV}=cuda but CUDA is unusable; using the CPU")
    _DEVICE_CACHE.append(device)
    return device


def reset_device_cache():
    """Forget the cached device (tests)."""
    _DEVICE_CACHE.clear()
