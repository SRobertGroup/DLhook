"""Super-resolution is opt-in in the GUI (utils/superres_flag.py)."""
import re
from pathlib import Path

from utils.superres_flag import ENV_VAR, superres_requested


def test_off_unless_asked_for():
    assert superres_requested({}) is False
    assert superres_requested({ENV_VAR: ""}) is False and superres_requested({ENV_VAR: "0"}) is False
    assert all(superres_requested({ENV_VAR: v}) for v in ("1", "true", "YES", " on "))


def test_the_gui_does_not_import_or_build_realesrgan_by_default():
    src = Path(__file__).resolve().parent.parent.joinpath("seedling_measurement.py").read_text(encoding="utf-8")
    top = src.split("class Gui", 1)[0]
    assert "RealesrganSuperresolution" not in top                        # no import at module load
    assert re.search(r"if self\.model_superres is not None:", src)       # the crop step is gated on the opt-in model
    assert "if torch.cuda.is_available():\n                    # Super-resolution" not in src.replace("\r\n", "\n")
