"""Download the pre-trained DLhook weights from their Zenodo deposit.

    python -m utils.fetch_weights            # the files the default pipeline needs
    python -m utils.fetch_weights --all      # also the fallback and legacy files

The deposit is one zip (`DLHOOK_WEIGHTS_ZIP`); every file is checked against the SHA-256 pinned
below before it is written, and an existing file is never replaced, so a locally fine-tuned
network wins. Nothing is downloaded at start-up: a missing file still triggers the documented
fallback with a warning. Pure stdlib; `--url` may be a file:// URL (the tests use that).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tempfile
import urllib.request
import zipfile

from utils.paths import repo_path

log = logging.getLogger(__name__)

ZENODO_DOI = "10.5281/zenodo.23531884"
ZENODO_RECORD = ZENODO_DOI.rsplit(".", 1)[-1]
WEIGHTS_ZIP = "DLhook_weights_v1.0.zip"
WEIGHTS_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{WEIGHTS_ZIP}?download=1"
ZIP_PREFIX = "DLhook_weights_v1.0/"

# repo-relative path -> SHA-256 (the deposit's SHA256SUMS.txt)
REQUIRED = {
    "weights/landmarks/dlhook_landmarks_m_a.pt": "0dc6515c8c8c8999427038633eeba373995ec447d4cbb0922be9de9f3956543b",
    "weights/landmarks/dlhook_landmarks_m_b.pt": "c8589adece2307cf78a3da87a85ebee04e5f335e8a58b4b12a97e43a1d608817",
    "weights/landmarks/dlhook_landmarks_m_c.pt": "dfe80e420f95a5374f54f2ac7903514ea05fc1414403071580a68d0ac60487f7",
    "weights/multiclass/dlhook_4class_v2.pt": "2608ff2025b9abcb4a31516afd55812e7475aedecee60a8fe51e0b5e2942a847",
    "weights/multiclass/dlhook_4class_v1.pt": "acd8048ab61873ba22c954dba4a1c76439cbfc66877d5d72817b89c382083f2b",
    "weights/germination_detector.json": "086af0a780559f94f25f248b5e906a9ba870d59406e52ea3909da856448b3aed",
}
OPTIONAL = {
    "weights/landmarks/dlhook_landmarks_a.pt": "bef420510f780c79fc202bfe552902e93d959ebfb451cc01248405daa7143235",
    "weights/landmarks/dlhook_landmarks_b.pt": "5a774e21204881b057af7d81fc7728e885647bdc75f33b9f7758998604c7265d",
    "weights/landmarks/dlhook_landmarks_c.pt": "bec84903bff176fa6139faf77cbaa8135a133c97a08d32da8fc0c26c8e54aa4a",
    "weights/RootPainter_weights/cotyledon_v5.pkl": "4d3977b242ecf6ef91022f579f036d914e89e535b4675243467cb95de3cc2faa",
    "weights/RootPainter_weights/hypocot_v5.pkl": "7b00bbf309efeef0a1d58542d5b6df42d9fb75fac50cc069c04ddace286d3b48",
    "weights/RootPainter_weights/germ_v1.pkl": "677ac182ceb2c92c80d2704539b4717fee19f81ce37078780734f8800fa1bdff",
    "weights/RootPainter_weights/germ_v2.pkl": "390219cabf5a7dcd4cb39ca8f91ecf9e29deef822ebedf1865e9a75b29c36937",
}


def _sha256(data_iter) -> str:
    h = hashlib.sha256()
    for block in data_iter:
        h.update(block)
    return h.hexdigest()


def fetch_weights(url: str = WEIGHTS_URL, include_optional: bool = False, root=None) -> dict:
    """Install the missing files of the deposit under `root` (default: the repository).

    Returns {"installed": [...], "present": [...]} (repo-relative paths). Raises RuntimeError
    when a file in the archive does not match its pinned SHA-256 (nothing of it is installed)
    or is missing from the archive."""
    wanted = dict(REQUIRED, **(OPTIONAL if include_optional else {}))
    dest = (lambda rel: os.path.join(root, rel)) if root else repo_path
    present = [rel for rel in wanted if os.path.exists(dest(rel))]
    todo = [rel for rel in wanted if rel not in present]
    if not todo:
        return {"installed": [], "present": present}

    log.info(f"Downloading the DLhook weights (doi:{ZENODO_DOI}) from {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, WEIGHTS_ZIP)
        with urllib.request.urlopen(url, timeout=60) as resp, open(archive, "wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        installed = []
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            for rel in todo:
                member = ZIP_PREFIX + rel
                if member not in names:
                    raise RuntimeError(f"{rel} is not in the downloaded archive")
                with zf.open(member) as fh:
                    digest = _sha256(iter(lambda: fh.read(1 << 20), b""))
                if digest != wanted[rel]:
                    raise RuntimeError(f"{rel}: SHA-256 mismatch (got {digest}, expected {wanted[rel]}); "
                                       "nothing of it was installed")
                target = dest(rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                part = target + ".part"
                with zf.open(member) as fh, open(part, "wb") as out:
                    while True:
                        block = fh.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                os.replace(part, target)
                installed.append(rel)
    return {"installed": installed, "present": present}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="also the fallback and legacy files")
    parser.add_argument("--url", default=WEIGHTS_URL, help="the deposit zip (default: Zenodo)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = fetch_weights(args.url, include_optional=args.all)
    except Exception as exc:
        print(f"Could not fetch the weights: {exc}", file=sys.stderr)
        return 1
    for rel in result["installed"]:
        print(f"installed  {rel}")
    for rel in result["present"]:
        print(f"kept       {rel} (already there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
