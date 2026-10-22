import pytest

from multi.src.config import load_config, resolved_path


def _write_config(path, data_num_classes=4, model_num_classes=4):
    path.write_text(
        f"""
paths:
  raw_data_dir: training_dataset/dlhook
  absolute_thing: "C:/some/absolute/path"
data:
  num_classes: {data_num_classes}
model:
  num_classes: {model_num_classes}
""",
        encoding="utf-8",
    )


def test_mismatched_num_classes_raises_a_clear_error(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    _write_config(cfg_path, data_num_classes=4, model_num_classes=2)

    with pytest.raises(ValueError, match="num_classes"):
        load_config(cfg_path)


def test_matching_num_classes_loads_fine(tmp_path):
    cfg_path = tmp_path / "good.yaml"
    _write_config(cfg_path, data_num_classes=4, model_num_classes=4)

    config = load_config(cfg_path)
    assert config["data"]["num_classes"] == 4


def test_resolved_path_uses_explicit_data_root_over_repo_root(tmp_path):
    cfg_path = tmp_path / "good.yaml"
    _write_config(cfg_path)

    config = load_config(cfg_path, data_root=tmp_path)
    assert resolved_path(config, "raw_data_dir") == tmp_path / "training_dataset" / "dlhook"


def test_resolved_path_leaves_an_absolute_path_untouched(tmp_path):
    cfg_path = tmp_path / "good.yaml"
    _write_config(cfg_path)

    config = load_config(cfg_path, data_root=tmp_path)
    resolved = resolved_path(config, "absolute_thing")
    assert str(resolved) in ("C:\\some\\absolute\\path", "C:/some/absolute/path")
