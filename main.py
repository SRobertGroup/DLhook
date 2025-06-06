import tkinter as tk
from seedling_measurment import *


def main():
    remove_files_exit()
    root = tk.Tk()
    Gui(root)
    root.geometry("1280x700+0+0")
    root.mainloop()

if __name__ == "__main__":
    main()
