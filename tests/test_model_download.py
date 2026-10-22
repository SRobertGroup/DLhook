import hashlib

import pytest

from utils.model_download import ensure_file, sha256_of


def test_downloads_once_checks_the_hash_and_never_replaces_an_existing_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"weights" * 1000)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "sub" / "w.pth"
    assert ensure_file(dest, src.as_uri(), digest) == str(dest) and sha256_of(dest) == digest
    assert not (tmp_path / "sub" / "w.pth.part").exists()
    src.write_bytes(b"different")                        # a later source change must not touch an installed file
    ensure_file(dest, src.as_uri(), digest)
    assert sha256_of(dest) == digest


def test_a_bad_download_installs_nothing(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"corrupted")
    dest = tmp_path / "w.pth"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ensure_file(dest, src.as_uri(), "0" * 64)
    assert not dest.exists() and not (tmp_path / "w.pth.part").exists()
    with pytest.raises(Exception):
        ensure_file(dest, (tmp_path / "missing.bin").as_uri(), "0" * 64)
    assert not dest.exists() and not (tmp_path / "w.pth.part").exists()


def test_superres_weights_info_pins_the_known_hash():
    # constants only: the wrapper itself imports basicsr, which needs the torchvision shim and a PIL import order
    from models.superres import weights_info as wi
    import os
    assert wi.WEIGHTS_PATH.replace("\\", "/").endswith("models/superres/RealESRGAN_x4plus.pth")
    assert len(wi.WEIGHTS_SHA256) == 64 and wi.WEIGHTS_URL.startswith("https://github.com/xinntao/Real-ESRGAN/")
    if os.path.exists(wi.WEIGHTS_PATH):                    # local checkouts: the pinned hash matches the shipped file
        assert sha256_of(wi.WEIGHTS_PATH) == wi.WEIGHTS_SHA256
