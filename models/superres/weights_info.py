"""Where the Real-ESRGAN x4plus weights live and how they are fetched (pure constants, no heavy imports).

The 64 MB file is not tracked in git (docs/AUDIT.md H-6). `RealesrganSuperresolution` downloads it
from the official Real-ESRGAN release on first use and checks the SHA-256 below.
"""
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RealESRGAN_x4plus.pth")
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
