import os

# Cleans specific folders on exit from the seedling GUI
class RemoveData:
    def __init__(self):
        self.paths = [
            'data/images/',
            'data/predict/',
            'data/postprocess/',
            'data/final_prediction/'
        ]

        for folder in self.paths:
            os.makedirs(folder, exist_ok=True)  # Ensure folder exists

            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"[ERROR] Could not delete {file_path}: {e}")

if __name__ == '__main__':
    cleaner = RemoveData()
