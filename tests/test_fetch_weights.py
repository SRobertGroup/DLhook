import hashlib
import zipfile

import pytest

from utils import fetch_weights as fw


def _deposit(tmp_path, files):
    """A zip laid out like the Zenodo deposit; returns (file:// url, {rel: sha256})."""
    archive = tmp_path / fw.WEIGHTS_ZIP
    with zipfile.ZipFile(archive, "w") as zf:
        for rel, data in files.items():
            zf.writestr(fw.ZIP_PREFIX + rel, data)
    return archive.as_uri(), {rel: hashlib.sha256(d).hexdigest() for rel, d in files.items()}


def test_pinned_manifest_matches_the_deposit_layout():
    assert len(fw.REQUIRED) == 6 and len(fw.OPTIONAL) == 7
    assert not set(fw.REQUIRED) & set(fw.OPTIONAL)
    assert all(len(h) == 64 for h in {**fw.REQUIRED, **fw.OPTIONAL}.values())
    assert fw.ZENODO_RECORD in fw.WEIGHTS_URL and fw.WEIGHTS_URL.endswith("?download=1")


def test_installs_missing_files_keeps_existing_ones_and_skips_optional(tmp_path, monkeypatch):
    files = {"weights/a.pt": b"a" * 100, "weights/b.pt": b"b" * 100, "weights/opt.pt": b"o" * 100}
    url, sums = _deposit(tmp_path, files)
    monkeypatch.setattr(fw, "REQUIRED", {k: sums[k] for k in ("weights/a.pt", "weights/b.pt")})
    monkeypatch.setattr(fw, "OPTIONAL", {"weights/opt.pt": sums["weights/opt.pt"]})
    root = tmp_path / "repo"
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "a.pt").write_bytes(b"locally fine-tuned")

    result = fw.fetch_weights(url, root=str(root))
    assert result == {"installed": ["weights/b.pt"], "present": ["weights/a.pt"]}
    assert (root / "weights" / "a.pt").read_bytes() == b"locally fine-tuned"      # never replaced
    assert (root / "weights" / "b.pt").read_bytes() == files["weights/b.pt"]
    assert not (root / "weights" / "opt.pt").exists()
    assert fw.fetch_weights(url, root=str(root))["installed"] == []               # nothing left: no download

    assert fw.fetch_weights(url, include_optional=True, root=str(root))["installed"] == ["weights/opt.pt"]


def test_a_corrupted_or_incomplete_archive_installs_nothing(tmp_path, monkeypatch):
    url, sums = _deposit(tmp_path, {"weights/a.pt": b"corrupted"})
    root = tmp_path / "repo"
    monkeypatch.setattr(fw, "REQUIRED", {"weights/a.pt": "0" * 64})
    monkeypatch.setattr(fw, "OPTIONAL", {})
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fw.fetch_weights(url, root=str(root))
    assert not (root / "weights" / "a.pt").exists() and not list(root.rglob("*.part"))

    monkeypatch.setattr(fw, "REQUIRED", {"weights/missing.pt": sums["weights/a.pt"]})
    with pytest.raises(RuntimeError, match="not in the downloaded archive"):
        fw.fetch_weights(url, root=str(root))
