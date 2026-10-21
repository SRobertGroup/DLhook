"""Shared test setup."""
import pytest

from utils import paths


@pytest.fixture(autouse=True)
def _session_data_in_tmp(tmp_path, monkeypatch):
    """Session scratch files (crops, mask dumps) go to this test's tmp_path/data, never the repo's
    data/ folder. Tests that chdir into tmp_path and write data/images there see the same files."""
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
