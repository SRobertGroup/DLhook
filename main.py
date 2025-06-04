import tkinter as tk
from seedling_measurment import *
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch
print("torch has access to GPUs Available: ", torch.cuda.is_available())

def main():
    remove_files_exit()
    root = tk.Tk()
    Gui(root)
    root.geometry("1280x700+0+0")
    root.mainloop()

if __name__ == "__main__":
    main()


# ---------------
## Archive 
# ---------------
# from seedling_measurment import *
# 
# remove_files_exit()
# root=tk.Tk()
# gui=Gui(root)
# w, h = root.winfo_screenwidth(), root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (1280, 700))
# root.mainloop()