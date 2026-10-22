"""The dependency files stay pinned and agree with each other (docs/AUDIT.md H-4)."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.!+\-]+)$")


def pins(name):
    out = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = PIN.match(line)
        assert m, f"{name}: {line!r} is not an exact == pin"
        out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def public(version):
    return version.split("+")[0]


def test_every_runtime_dependency_is_pinned_exactly():
    runtime = pins("requirements.txt")
    for needed in ("torch", "torchvision", "numpy", "pandas", "pillow", "matplotlib", "scikit-image",
                   "opencv-python", "pyyaml", "tqdm", "basicsr", "realesrgan"):
        assert needed in runtime, f"requirements.txt does not pin {needed}"
    assert "torchaudio" not in runtime            # nothing imports it


def test_lock_and_ci_agree_with_the_runtime_pins():
    runtime, lock, ci = pins("requirements.txt"), pins("requirements-lock.txt"), pins("requirements-ci.txt")
    for name, version in runtime.items():
        assert public(lock[name]) == version, f"lock has {name} {lock[name]}, requirements.txt {version}"
        if name in ci:
            assert ci[name] == version, f"requirements-ci.txt has {name} {ci[name]}, requirements.txt {version}"
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev and public(pins("requirements-dev.txt")["pytest"]) == ci["pytest"]


def test_ci_torch_is_the_pinned_torch():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    runtime = pins("requirements.txt")
    assert f"torch=={runtime['torch']}" in workflow and f"torchvision=={runtime['torchvision']}" in workflow


def test_env_yaml_installs_the_pinned_requirements():
    env = yaml.safe_load((ROOT / "env.yaml").read_text(encoding="utf-8"))
    pip_section = next(d["pip"] for d in env["dependencies"] if isinstance(d, dict))
    assert "-r requirements.txt" in pip_section
    assert any("download.pytorch.org/whl/cu" in p for p in pip_section)
    assert "python=3.10" in env["dependencies"] and "torchaudio" not in str(env)
