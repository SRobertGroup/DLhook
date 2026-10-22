import logging
import os
import tkinter as tk

from seedling_measurement import Gui, remove_files_exit


def main():
    # DLHOOK_LOG_LEVEL=DEBUG shows the per-frame details (contours, angles); WARNING keeps only problems.
    logging.basicConfig(level=os.environ.get("DLHOOK_LOG_LEVEL", "INFO").upper(),
                        format="%(levelname)s %(name)s: %(message)s")
    remove_files_exit()
    root = tk.Tk()
    Gui(root)
    root.geometry("1280x710+0+0")
    root.mainloop()

if __name__ == "__main__":
    main()
