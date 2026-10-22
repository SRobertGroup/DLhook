"""Where DLhook's files live, independent of the working directory.

Weight files are named as repo-relative strings (models/segmentation_backends.py); `repo_path`
resolves such a string against the repository root at the point a file is opened or checked, so
launching `python <repo>/main.py` from any directory finds the same weights. Absolute paths pass
through unchanged.

The session scratch folders (crops, optional mask dumps; wiped by utils/clean_on_exit.RemoveData)
live under DATA_ROOT, by default <repo>/data, or DLHOOK_DATA_DIR when set. `data_path` reads
DATA_ROOT at call time, so tests can point it at a temporary folder.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DLHOOK_DATA_DIR") or REPO_ROOT / "data")


def repo_path(path):
    """`path` resolved against the repository root (absolute paths are returned as they are)."""
    path = Path(path)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def data_path(*parts, make=False):
    """A path under DATA_ROOT, e.g. data_path("images", name). With make=True the parent folder
    of the returned path (or the path itself when no file name is given) is created."""
    path = Path(DATA_ROOT, *parts)
    if make:
        (path if not parts or not Path(parts[-1]).suffix else path.parent).mkdir(parents=True, exist_ok=True)
    return str(path)


def images_dir():
    """The folder holding the current session's crops."""
    return data_path("images")


def crop_path(file_name):
    """Full path of a crop file name ("{crop_id}-crop-{stem}.png")."""
    return data_path("images", file_name)
