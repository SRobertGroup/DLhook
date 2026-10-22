"""Paths never depend on the working directory, and image extensions of any length work (audit M-5)."""
import os
from pathlib import Path

from utils import paths


def test_repo_path_resolves_against_the_repo_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(paths.repo_path("weights/x.pt")) == paths.REPO_ROOT / "weights" / "x.pt"
    absolute = str(tmp_path / "elsewhere.pt")
    assert paths.repo_path(absolute) == absolute


def test_data_path_follows_data_root_and_can_create_folders(tmp_path):
    assert Path(paths.crop_path("0-crop-a.png")) == tmp_path / "data" / "images" / "0-crop-a.png"
    target = paths.data_path("predict", "m.png", make=True)
    assert (tmp_path / "data" / "predict").is_dir() and not os.path.exists(target)
    assert os.path.isdir(paths.data_path("final_prediction", make=True))


def test_default_backend_finds_its_weights_from_any_cwd(tmp_path, monkeypatch):
    from models import segmentation_backends as sb

    repo = tmp_path / "repo"
    ckpt = repo / sb.DEFAULT_MULTICLASS_CHECKPOINT
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.delenv(sb.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)                       # not the repo: a cwd-relative check would miss it
    assert sb.resolve_backend_name() == "multiclass"


def test_remove_data_only_touches_data_root(tmp_path, monkeypatch):
    from utils.clean_on_exit import RemoveData

    cwd = tmp_path / "somewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    stale = tmp_path / "data" / "images" / "old.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    RemoveData()
    assert not stale.exists() and (tmp_path / "data" / "predict").is_dir()
    assert not (cwd / "data").exists()


def test_long_and_upper_case_extensions_are_listed_and_named_by_stem(tmp_path):
    import seedling_measurement as sm

    for name in ("a.tiff", "b.JPEG", "c.jpg", "d.TIF", "notes.csv"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.png").mkdir()
    assert sm._list_image_files(str(tmp_path)) == ["a.tiff", "b.JPEG", "c.jpg", "d.TIF"]
    assert [f"0-crop-{sm._stem(n)}.png" for n in ("a.tiff", "b.JPEG")] == ["0-crop-a.png", "0-crop-b.png"]
