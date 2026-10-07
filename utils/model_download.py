"""Fetch a large model file on first use instead of tracking it in git.

`ensure_file(path, url, sha256)` returns `path` if it exists (an existing file is never replaced, so
a locally fine-tuned or manually placed file wins); otherwise it downloads `url` next to it,
checks the SHA-256 and moves it into place, so an interrupted or corrupted download never leaves a
file that looks valid. Pure stdlib; `url` may be file:// (the tests use that).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


def sha256_of(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ensure_file(path, url: str, sha256: str, label: str = "model weights") -> str:
    path = Path(path)
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"Downloading {label} to {path} (first use only) from {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        digest = sha256_of(part)
        if digest != sha256:
            raise RuntimeError(f"{label}: SHA-256 mismatch for {url} (got {digest}, expected {sha256}); "
                               f"nothing was installed")
        os.replace(part, path)
    except Exception:
        if part.exists():
            part.unlink()
        raise
    return str(path)
