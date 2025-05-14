from seedling_measurment import *

remove_files_exit()
root=tk.Tk()
gui=Gui(root)
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (1280, 700))
root.mainloop()