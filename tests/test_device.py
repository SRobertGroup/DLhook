"""utils/device.py: a CUDA device is used only after a test op succeeds."""
import pytest
import torch

from utils import device as dev


@pytest.fixture(autouse=True)
def _fresh_cache():
    dev.reset_device_cache()
    yield
    dev.reset_device_cache()


def test_cpu_when_cuda_is_not_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.pick_device(environ={}).type == "cpu"


def test_falls_back_to_cpu_when_the_cuda_probe_fails(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_ones = torch.ones

    def ones(*args, device=None, **kwargs):
        if device == "cuda":
            raise RuntimeError("no kernel image is available for execution on the device")
        return real_ones(*args, device=device, **kwargs)

    monkeypatch.setattr(torch, "ones", ones)
    assert dev.pick_device(environ={}).type == "cpu"
    assert any("test operation failed" in r.getMessage() for r in caplog.records)


def test_env_override_to_cpu_skips_the_probe(monkeypatch):
    def boom():
        raise AssertionError("probe must not run")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert dev.pick_device(environ={"DLHOOK_DEVICE": "cpu"}).type == "cpu"


def test_unknown_env_value_raises():
    with pytest.raises(ValueError):
        dev.pick_device(environ={"DLHOOK_DEVICE": "tpu"})


def test_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: calls.append(1) or False)
    first = dev.pick_device(environ={})
    assert dev.pick_device(environ={}) is first and len(calls) == 1
