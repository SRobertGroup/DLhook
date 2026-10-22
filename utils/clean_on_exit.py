import logging
import os

from utils.paths import data_path

log = logging.getLogger(__name__)

# Cleans specific folders on exit from the seedling GUI.
#
# Masks now live in an in-memory MaskStore (utils/mask_store.py) instead of
# being round-tripped through data/predict/ + data/postprocess/, so those two
# directories are normally empty -- they are still listed here because the
# opt-in DLHOOK_DUMP_MASKS dump writes into data/predict/, and because this
# class is also what CREATES all four directories (see the makedirs below,
# which other code depends on for data/images/).
#
# The folders are under utils.paths.DATA_ROOT (<repo>/data by default), never the working
# directory, so launching from elsewhere cannot create or empty a stray data/ folder.
SCRATCH_FOLDERS = ("images", "predict", "postprocess", "final_prediction")


class RemoveData:
    def __init__(self):
        self.paths = [data_path(name) for name in SCRATCH_FOLDERS]

        for folder in self.paths:
            os.makedirs(folder, exist_ok=True)  # Ensure folder exists

            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        log.error(f"Could not delete {file_path}: {e}")

if __name__ == '__main__':
    cleaner = RemoveData()
