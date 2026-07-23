import os
import tkinter as tk
import numpy as np
import pandas as pd
import shutil
import atexit
import sys

# from subprocess import list2cmdline
from tkinter.ttk import Progressbar, Style
from tkinter.filedialog import asksaveasfilename, askdirectory
from tkinter import simpledialog
from models.UNetInference import *  # RootPainter
# from numpy.lib.function_base import select
from utils.apicalhook_angle import *
from utils.clean_on_exit import *
from utils.matching_crop2points_GUI import *
from utils.preprocess_model_input import *
from utils.postprocmask import PostprocessMasks, point_num
from utils.gui_thread_safety import ProgressReporter
from utils.germination_detector import GerminationDetector
from ui.analysis_window import SeedlingAnalysisWindow

from datetime import datetime
import threading
import json

import re

import ast
from PIL import Image, ImageTk

try:
    resample_filter = Image.Resampling.LANCZOS  # Pillow ≥ 10
except AttributeError:
    resample_filter = Image.ANTIALIAS  # Pillow < 10

import cv2

import torch
from torchvision.transforms import functional
import sys

sys.modules["torchvision.transforms.functional_tensor"] = functional

#Superres libraries 
from models.superres.superresolution_predict import RealesrganSuperresolution


"""
This is the graphical user interface for the DLhook software

"""
class Gui():
    def __init__(self, root):

       
        self.model_superres=RealesrganSuperresolution()
        sys.setrecursionlimit(10000)


        self.path_crop='data/images/'
        self.path_save_x='data/images/'


        self.angles=[]

        # Seedling start/end point pairs (image coords): {"start": (x,y), "end": (x,y)}
        self.seedling_pairs=[]
        self.selected_points_debug=[]

        self.remove_regions=[]

        self.crop_range_points=[]

        self.img_count=0
        self.photo_list=[]
        self.metadata_list=[]
        self.fallback_interval_minutes=None
        self.time_deltas=[]

        # Per-seedling adjustable crop boxes (image coords), 1:1 with seedling_pairs,
        # auto-derived from each pair's bounding box: {"cx","cy","half_w","half_h"}
        self.crop_boxes=[]
        self._pending_start=None
        self._box_canvas_items=[]
        self._point_canvas_items=[]
        self._active_handle=None

        self.transformed_mid_points=[]
        self.crop_points_distributed=[]
        self.crop_points_numberID=[]

        # Per-seedling germination time-zero (crop_id -> frame index or None),
        # detected from the germ_v1 mask time series; user-overridable (Phase 5).
        self.germination_detector=None

        # Per-seedling (crop_id) time series of cotyledon/germ contours, one
        # entry appended per frame by process_single_frame -- keyed by crop_id
        # so the batch pipeline and the Phase 5 per-seedling preview window
        # can share the exact same per-frame logic without interleaving
        # each other's history.
        self.cotyl_time_series_by_crop={}
        self.germ_time_series_by_crop={}

        self.activ_button=False

        #The following variables are used to controll the buttons,
        self.check_place_rect=False
        self.crop_window_check=False
        self.rectangle_circle_match=[]
        self.manual_angle=False
        self.point_numb=1
        self.button_circle_check=False
        self.show_image_check=False
        self.button_rect_check=False
        self.rectangle_width=0
        self.rectangle_height=0

        self.crop_size = 1024
        self.crop_half_size = 512  # Half of the 1024x1024 crop area; also the default floor for auto-derived boxes

        self.x_length=1100
        self.y_length=650
        width=self.x_length
        height=self.y_length
        
        self.filenames_listbox=False

        self.root=root

        # configure style
        self.style = Style(self.root)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        icon_img = ImageTk.PhotoImage(file="data/logo/icon.png")
        self.icon_img = icon_img
        self.root.iconphoto(False, self.icon_img)
        # self.root.iconbitmap("@data/logo/icon.xbm")
        self.root.title('DLhook')
        self.root.geometry(str(self.x_length) + "x" + str(self.y_length))
        self.fin = tk.Frame(self.root, width=200, height=200)
        self.fin.pack()
        self.fin.place(x=0, y=0)
        self.canvas = tk.Canvas(self.fin, bg='#FFFFFF', width=width, height=height)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.btn_import_image = tk.Button(self.root, text="Open image directory", width=17, command=self.add_files)

        self.btn_import_image.place(x=width + 25, y=height - 640)
        
        self.listbox = tk.Listbox(self.root, width=25)
        self.listbox.place(x=width + 10, y=height - 610)

        self.btn_show_image = tk.Button(self.root, text="Show image", width=10, command=self.show_image)
        self.btn_show_image["state"]=tk.DISABLED
        self.btn_show_image.place(x=width + 50, y=height - 440)

        self.label_user_input = tk.Label(self.root, text = "User Input")
        self.label_user_input.place(x=width + 59, y=height - 305)

        self.label_1 = tk.Label(self.root, text = "1.")
        self.label_1.place(x=width + 20, y=height - 277)
        self.button_place_point=tk.Button(self.root, text="Place points", width=10, command=self.canvas_circle_activate)
        self.button_place_point.place(x=width + 50, y=height - 280)
        self.button_place_point["state"]=tk.DISABLED

        self.label_2 = tk.Label(self.root, text = "2.")
        self.label_2.place(x=width + 20, y=height - 227)
        self.button_crop = tk.Button(self.root, text="Crop image", width=10, command=self.crop_image)
        self.button_crop.place(x=width + 50, y=height - 230)
        self.button_crop["state"]=tk.DISABLED

        self.crop_padding_var = tk.StringVar(value="40")
        self.crop_padding_label = tk.Label(self.root, text="Box padding %:")
        self.crop_padding_label.place(x=width + 20, y=height - 130)
        self.crop_padding_entry = tk.Entry(self.root, width=5, textvariable=self.crop_padding_var)
        self.crop_padding_entry.place(x=width + 140, y=height - 130)


        self.label_3 = tk.Label(self.root, text = "3.")
        self.label_3.place(x=width + 20, y=height - 177)
        self.button_start_analysis = tk.Button(self.root, text="Start Analysis", width=10, command=self._threading_analysis)
        self.button_start_analysis.place(x=width + 50, y=height - 180)
        self.button_start_analysis["state"]=tk.DISABLED

        self.label_4 = tk.Label(self.root, text = "4.")
        self.label_4.place(x=width + 20, y=height - 80)
        self.button_preview_seedlings = tk.Button(self.root, text="Preview seedling", width=13, command=self.open_seedling_picker)
        self.button_preview_seedlings.place(x=width + 50, y=height - 83)
        self.button_preview_seedlings["state"]=tk.DISABLED


        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=700)
        self.progress.place(x=width - 900, y=height + 15)
        self.progress_bar_label = tk.Label(self.root,
                  text = "Measurement progress:")
        self.progress_bar_label.place(x=width - 1030, y=height + 15)

        # Thread-safe channel for background analysis to report progress back to the GUI
        self.progress_reporter = ProgressReporter()
        self.root.after(100, self._pump_progress)


        # --- Load and display the logo ---
        logo_image = Image.open("data/logo/logos.png")
        logo_image = logo_image.resize((140, 117), resample_filter)  # adjust size if needed
        self.logo_photo = ImageTk.PhotoImage(logo_image)  # keep a reference

        self.logo_label = tk.Label(self.root, image=self.logo_photo, borderwidth=0)
        self.logo_label.place(x=width + 30, y=height - 70)  # adjust as needed

        self.debug_var=tk.IntVar(value=1)
        self.check_button_debug= tk.Checkbutton(self.root, text='Debug Mode',variable=self.debug_var)
        self.check_button_debug.place(x=width + 45, y=height + 18)
        #self.button_place_point_remove=tk.Button(self.root, text="Points remove section", width=20, command=self.replace_angle)
        #self.button_place_point_remove.place(x=width + 10, y=height - 260)

        self.angle_p=[]
        self.angle_lines=[]
        self.new_angle_points=[]
        self.line1=False
        self.vinkel=[]


    def start_sort(self):
        """Record the start index based on selection length."""
        try:
            self.str1 = len(self.sort_txt_box.selection_get())
            print(f"[DEBUG] Start string length (str1): {self.str1}")
        except tk.TclError:
            self.str1 = 0
            print("[DEBUG] No selection made for start string.")

    def end_sort(self):
        """Sort files by number extracted from filename."""
        print("[DEBUG] Starting end_sort...")

        sorted_filenames = []
        for file in self.file_list:
            try:
                # Remove file extension
                base_name = os.path.splitext(file)[0]
                # Extract number from entire base name
                number_match = re.search(r'\d+', base_name)
                if number_match:
                    sort_key = int(number_match.group())
                    sorted_filenames.append((sort_key, file))
                else:
                    print(f"[DEBUG] No number found in: {file}")
            except Exception as e:
                print(f"[DEBUG] Error processing {file}: {e}")

        sorted_filenames.sort(key=lambda x: x[0])
        sorted_filenames_output = [file[1] for file in sorted_filenames]

        print("[DEBUG] Sorted filenames:")
        for f in sorted_filenames_output:
            print(f)

        self.add_files(sorted_filenames_output)
        tk.messagebox.showinfo("Sorting Complete", "Image files have been successfully sorted.")

    def imagename_in_sort_section(self):
        """Populate text box with the selected filename to extract sorting substring."""
        # self.sort_start_b["state"] = tk.NORMAL
        # self.sort_end_b["state"] = tk.NORMAL

        clicked_file = self.listbox.curselection()
        for item in clicked_file:
            self.selected_image = self.listbox.get(item)
            self.sort_txt_box.delete("1.0", tk.END)  # Ensure clean state
            self.sort_txt_box.insert(tk.INSERT, self.selected_image)
            print(f"[DEBUG] Selected image for sorting reference: {self.selected_image}")

    def replace_angle(self):
        if self.activ_button:
            self.activate_manual_ang()

        file_name = self.images_final[self.n_angle_image]
        org_filename = self.convert_filename(file_name)

        clicked_file = self.listbox.curselection()
        for item in clicked_file:
            data = self.listbox.get(item).split()
            if len(data) < 2:
                print(f"[DEBUG] Malformed listbox entry: {data}")
                return

            id_n = int(data[0])
            new_angle = self.new_angle

            try:
                row_id = int(self.new_df[self.new_df['img_name'] == org_filename].index.values[0])
                self.new_df.at[row_id, id_n] = new_angle
                print(f"[DEBUG] Updated angle for seedling {id_n} to {new_angle}")
            except Exception as e:
                print(f"[DEBUG] Error updating angle: {e}")
                return

        # Refresh listbox using the unified method
        self.update_listbox_for_angles(file_name)

    def blank_angle(self):
        file_name = self.images_final[self.n_angle_image]
        org_filename = self.convert_filename(file_name)

        clicked_file = self.listbox.curselection()
        for item in clicked_file:
            data = self.listbox.get(item).split()
            if len(data) < 2:
                print(f"[DEBUG] Malformed listbox entry: {data}")
                continue

            id_n = int(data[0])  # seedling ID

            try:
                row_id = int(self.new_df[self.new_df['img_name'] == org_filename].index.values[0])
                self.new_df.at[row_id, id_n] = np.nan  
                print(f"[DEBUG] Blank angle for seedling {id_n} at row {row_id}")
            except Exception as e:
                print(f"[DEBUG] Failed to blank angle for {id_n}: {e}")

        # Refresh GUI list
        self.update_listbox_for_angles(file_name)


    def place_angle_point(self, event):
        width=200
        height=200
        x1, y1 = event.x, event.y
        r=1

        self.root.bind("<Motion>", self.place_angle_line)
        if (x1 < self.x_length) and (y1 < self.y_length) and self.manual_angle==True:
            if self.check_place_angele == 3:
                self.angle_label.destroy()
                for n in self.angle_p:
                    self.canvas.delete(n)
                self.angle_p=[]

                for m in self.angle_lines:
                    self.canvas.delete(m)
                self.angle_lines = []
                self.check_place_angele = 0
                self.new_angle_points=[]
                self.vinkel=[]

            if self.check_place_angele == 2:
                (x0, y0) = self.points
                self.check_place_angele = self.check_place_angele + 1
                point3 = self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="#b87c84")
                line3 = self.canvas.create_line(x0, y0, x1, y1,fill="#b87c84")



                self.angle_lines.append(line3)

                self.angle_p.append(point3)
                self.points = (x1, y1)
                self.new_angle_points.append((x1, y1))
                self.c = np.array([x1, y1])

            if self.check_place_angele==1:

                (x0,y0)=self.points
                self.check_place_angele = self.check_place_angele + 1
                point2 = self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="green")
                line2 = self.canvas.create_line(x0,y0,x1,y1,fill="#b87c84")

                self.b=np.array([x1,y1])

                self.angle_lines.append(line2)
                self.angle_p.append(point2)
                self.points = (x1, y1)
                self.new_angle_points.append((x1, y1))

            if self.check_place_angele==0:

                self.check_place_angele=self.check_place_angele+1
                point1=self.canvas.create_rectangle(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="green")
                self.angle_p.append(point1)
                self.points = (x1, y1)
                self.new_angle_points.append((x1,y1))
                self.a = np.array([x1, y1])

            if self.check_place_angele == 3:

                ba=self.a - self.b
                bc=self.c - self.b
                cosine_ang=np.dot(ba, bc) /(np.linalg.norm(ba) * np.linalg.norm(bc))
                angle= np.arccos(cosine_ang)
                self.new_angle=round(np.degrees(angle))

                overhook=self.overhook_var.get()
                if overhook==1:
                    self.new_angle=-self.new_angle
                    self.angle_label=tk.Label(self.root, text =str(self.new_angle)+'°')
                    self.angle_label.place(x=width + 545, y=height + 115)
                else:
                    self.angle_label=tk.Label(self.root, text =str(self.new_angle)+'°'+' ('+str(180-self.new_angle)+'°)')
                    self.angle_label.place(x=width + 545, y=height + 115)

    def del_angle_line(self):
        self.canvas.delete(self.line1)

    def place_angle_line(self,event):
        if self.check_place_angele<3 and self.manual_angle==True:
            if self.line1!=False:
                self.del_angle_line()

            x1, y1 = event.x, event.y
            (x0, y0)=self.points
            self.line1=self.canvas.create_line(x0, y0, x1 ,y1, fill="#b87c84")

    def activate_manual_ang(self):
        if self.activ_button==True:
            self.buttonPlace_angle.configure(bg='white')
            for m in self.angle_lines:
                self.canvas.delete(m)



        if self.activ_button==False:
            self.buttonPlace_angle.configure(bg="#d78a5e")
            self.activ_button=True



        if self.manual_angle==False:
            self.angle_points=[]
            self.check_place_angele=0
            self.manual_angle=True
            self.root.bind("<Button-1>", self.place_angle_point)
        else:
            self.manual_angle=False
            for n in self.angle_p:
                self.canvas.delete(n)
            self.angle_p = []

            for m in self.angle_lines:
                self.canvas.delete(m)

    def canvas_circle_activate(self):

        if self.button_circle_check == False:
            self.button_circle_check = True
            self.button_place_point.configure(bg="#d78a5e")

            self.root.bind('<Button-1>', self.place_circle)
            self.root.bind('z', self._undo_last_point)
            self.root.bind('Z', self._undo_last_point)

        else:
            self.button_circle_check = False
            self.button_place_point.configure(bg="white")
            self.root.unbind('z')
            self.root.unbind('Z')

            # Discard an unfinished start point that never got a matching end point
            if self._pending_start is not None:
                self.canvas.delete(self._pending_start["oval"])
                self._pending_start = None

            if self.crop_boxes:
                self.button_crop["state"]=tk.NORMAL

            if self.debug_var.get() ==0:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            #save list with input points from user for faster debug while reruning application
                with open(f"data/debug_data/output/input_starting_points_saved{dt_string}.txt", "w") as output:
                    output.write(str(self.seedling_pairs))


    def place_circle(self, event):
        if not (self.button_circle_check and self.show_image_check):
            return
        x1, y1 = event.x, event.y
        if not (x1 < self.x_length and y1 < self.y_length):
            return

        r = 4
        px, py = self.canvas_to_image_coords(x1, y1)
        print(f"x: {x1}; y: {y1} --> x: {px}; y: {py}")

        if self._pending_start is None:
            # First click of the pair: the seed-coat start point
            oval = self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="#536791")
            self._pending_start = {"image": (px, py), "oval": oval}
            return

        # Second click of the pair: the end point completes this seedling
        end_oval = self.canvas.create_oval(x1 - r, y1 - r, x1 + r, y1 + r, stipple='', fill="#8a5eb8")
        start_point = self._pending_start["image"]
        start_oval = self._pending_start["oval"]
        self._pending_start = None

        self.seedling_pairs.append({"start": start_point, "end": (px, py)})
        self.selected_points_debug.append(start_point)
        self._point_canvas_items.append((start_oval, end_oval))

        self._add_crop_box(start_point, (px, py))

    def _undo_last_point(self, event=None):
        """Bound to the Z key while point-placement mode is active: undo the
        most recent click (a pending unfinished start point if there is one,
        else the last completed start/end pair and its auto-derived crop box)."""
        if not self.button_circle_check:
            return

        if self._pending_start is not None:
            self.canvas.delete(self._pending_start["oval"])
            self._pending_start = None
            return

        if not self.seedling_pairs:
            return

        self.seedling_pairs.pop()
        self.crop_boxes.pop()
        self._refresh_transformed_mid_points()

        start_oval, end_oval = self._point_canvas_items.pop()
        self.canvas.delete(start_oval)
        self.canvas.delete(end_oval)

        box_item = self._box_canvas_items.pop()
        self.canvas.delete(box_item["rect"])
        for handle_id in box_item["handles"].values():
            self.canvas.delete(handle_id)

        if not self.crop_boxes:
            self.button_crop["state"] = tk.DISABLED

    def _get_padding_fraction(self):
        try:
            return max(0.0, float(self.crop_padding_var.get())) / 100
        except (ValueError, AttributeError):
            return 0.4

    def _compute_crop_box(self, start, end):
        self.update_rectangle_size()
        padding = self._get_padding_fraction()

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        half_w = (max_x - min_x) * (1 + 2 * padding) / 2
        half_h = (max_y - min_y) * (1 + 2 * padding) / 2

        # Floor only: never smaller than the old fixed default. No ceiling --
        # a box must be free to grow past that default to fully contain a seedling.
        half_w = max(half_w, self.crop_half_size)
        half_h = max(half_h, self.crop_half_size)

        return {
            "cx": round((min_x + max_x) / 2),
            "cy": round((min_y + max_y) / 2),
            "half_w": round(half_w),
            "half_h": round(half_h),
        }

    def _add_crop_box(self, start, end):
        box = self._compute_crop_box(start, end)
        self.crop_boxes.append(box)
        self._draw_crop_box(len(self.crop_boxes) - 1)
        self._refresh_transformed_mid_points()

    def _refresh_transformed_mid_points(self):
        self.transformed_mid_points = [(box["cx"], box["cy"]) for box in self.crop_boxes]

    def _crop_box_canvas_rect(self, box):
        cx = box["cx"] * self.x_length / self.width1
        cy = box["cy"] * self.y_length / self.height1
        hw = box["half_w"] * self.x_length / self.width1
        hh = box["half_h"] * self.y_length / self.height1
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    def _draw_crop_box(self, idx):
        box = self.crop_boxes[idx]
        x1, y1, x2, y2 = self._crop_box_canvas_rect(box)
        rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, width=3, outline="#d78a5e")

        handle_r = 5
        corners = {"nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2)}
        handles = {}
        for name, (hx, hy) in corners.items():
            handles[name] = self.canvas.create_rectangle(
                hx - handle_r, hy - handle_r, hx + handle_r, hy + handle_r,
                fill="#d78a5e", outline="black"
            )
        self._box_canvas_items.append({"rect": rect_id, "handles": handles})

    def _redraw_crop_box(self, idx):
        box = self.crop_boxes[idx]
        x1, y1, x2, y2 = self._crop_box_canvas_rect(box)
        item = self._box_canvas_items[idx]
        self.canvas.coords(item["rect"], x1, y1, x2, y2)

        handle_r = 5
        corners = {"nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2)}
        for name, (hx, hy) in corners.items():
            self.canvas.coords(item["handles"][name], hx - handle_r, hy - handle_r, hx + handle_r, hy + handle_r)

    def _find_handle_at(self, x, y):
        for idx, item in enumerate(self._box_canvas_items):
            for name, handle_id in item["handles"].items():
                hx1, hy1, hx2, hy2 = self.canvas.coords(handle_id)
                if hx1 <= x <= hx2 and hy1 <= y <= hy2:
                    return idx, name
        return None

    def _start_drag_handle(self, event):
        if not (event.x < self.x_length and event.y < self.y_length):
            return
        self._active_handle = self._find_handle_at(event.x, event.y)

    def _drag_handle(self, event):
        if self._active_handle is None:
            return
        idx, name = self._active_handle
        box = self.crop_boxes[idx]

        canvas_x = min(max(event.x, 0), self.x_length - 1)
        canvas_y = min(max(event.y, 0), self.y_length - 1)
        ix, iy = self.canvas_to_image_coords(canvas_x, canvas_y)

        x1, y1 = box["cx"] - box["half_w"], box["cy"] - box["half_h"]
        x2, y2 = box["cx"] + box["half_w"], box["cy"] + box["half_h"]
        if name == "nw":
            x1, y1 = ix, iy
        elif name == "ne":
            x2, y1 = ix, iy
        elif name == "sw":
            x1, y2 = ix, iy
        elif name == "se":
            x2, y2 = ix, iy

        min_half = 10
        half_w = max(abs(x2 - x1) / 2, min_half)
        half_h = max(abs(y2 - y1) / 2, min_half)

        box["cx"] = round((x1 + x2) / 2)
        box["cy"] = round((y1 + y2) / 2)
        box["half_w"] = round(half_w)
        box["half_h"] = round(half_h)

        self._redraw_crop_box(idx)
        self._refresh_transformed_mid_points()

    def _end_drag_handle(self, event):
        self._active_handle = None


    def restart_program(self):
        shutil.rmtree(self.path, ignore_errors=True)
        for file in os.listdir('model_data/images/'):
            os.remove('model_data/images/')
        python = sys.executable
        os.execl(python, python, * sys.argv)
        
    def update_rectangle_size(self):
        # Default/floor crop half-size, derived from image width; used by
        # _compute_crop_box as the minimum size for an auto-derived crop box.
        if hasattr(self, 'width1') and hasattr(self, 'height1'):
            self.crop_half_size = round(self.width1 / 8)
            self.crop_size = self.crop_half_size * 2
        else:
            print("[WARNING] Image dimensions not set yet. Cannot update rectangle size.")

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates (scaled) to actual image coordinates."""
        if not hasattr(self, 'width1') or not hasattr(self, 'height1'):
            raise ValueError("Image dimensions not initialized.")
        x_mult = canvas_x / self.x_length
        y_mult = canvas_y / self.y_length
        return round(self.width1 * x_mult), round(self.height1 * y_mult)

    #Transform the angle to the correct format
    def _correct_angle(self, angle):
        angle=round(angle)
        if angle>90:
            angle=angle-90
        else:
            angle=angle+90
        return angle

    #function for multithreading analysis in background
    def _threading_analysis(self):
        # Disable all buttons while analysis is ongoing (done here, on the main thread,
        # since start_analysis itself runs on the background thread from its first line)
        self.button_crop["state"]=tk.DISABLED
        self.button_start_analysis["state"]=tk.DISABLED
        self.button_place_point["state"]=tk.DISABLED
        self.btn_show_image["state"]=tk.DISABLED
        self.btn_import_image["state"]=tk.DISABLED

        # Read image timestamps (and prompt for a fallback interval if needed) here,
        # on the main thread -- this may pop a modal dialog, which is only safe
        # before the background worker starts, not from inside it.
        self._collect_image_metadata()

        th = threading.Thread(target=self.start_analysis)
        th.start()

    def _collect_image_metadata(self):
        self.metadata_list = []
        for image in self.file_list:
            img_path = os.path.join(self.path, image)
            creation_time = get_image_creation_time(img_path)
            self.metadata_list.append({"filename": image, "creation_time": creation_time})

        if metadata_has_gaps(self.metadata_list):
            self.fallback_interval_minutes = simpledialog.askfloat(
                "Missing timestamp metadata",
                "Some images are missing reliable creation-time metadata.\n"
                "Enter the interval between images (in minutes):",
                parent=self.root,
                minvalue=0.0,
            )
        else:
            self.fallback_interval_minutes = None

        self.time_deltas = compute_time_deltas(self.metadata_list, self.fallback_interval_minutes)

    def _pump_progress(self):
        """Runs on the main thread; drains ProgressReporter events from the worker thread
        and applies them to Tk widgets here instead of mutating them cross-thread."""
        for event in self.progress_reporter.drain():
            if "value" in event:
                self.progress["value"] = event["value"]
            if "message" in event:
                self.progress_bar_label.configure(text=event["message"])
            if event.get("done"):
                self.openNewWindow()
        self.root.after(100, self._pump_progress)

    def _reset_main_window(self):
        a=1


    """
    Starts the analysis of the kinematics batch
    """
    def start_analysis(self):
        #
        if self.debug_var.get() ==1:
            try:
                txt_input_files=os.listdir('data/debug_data/input')
                if len(txt_input_files)!=0:
                    with open(os.path.join('data/debug_data/input',txt_input_files[0])) as input_points_txt:
                        input_points_txt=input_points_txt.readlines()
                        self.selected_points_debug=ast.literal_eval(input_points_txt[0])
            except Exception:
                print("no file found!")


        self.check_place_rect=False
        self.button_crop.configure(bg="white")

        # Crop boxes are 1:1 with seedling start points by construction, so no
        # spatial bucketing/matching is needed beyond pairing by index.
        match_p = MatchCropPoints(
            self.crop_boxes,
            [pair["start"] for pair in self.seedling_pairs],
            working_size=self.crop_size,
        )
        self.crop_points_distributed = match_p.return_crop_points()

        # Initialize preprocessing
        preprocess_init = preprocess_images()

        # Sort crop points in each image from left to right
        self.crop_points_distributed = [
            sorted(points_x, key=lambda x: x[0])
            for points_x in self.crop_points_distributed
        ]

        # Get numbered points
        get_point_numbering = point_num(self.crop_points_distributed)
        self.crop_points_numberID = get_point_numbering.point_numbering()

        # Progress bar setup
        n_images = len(self.file_list)

        # Process each image
        for image_n, image in enumerate(self.file_list):
            img_path = os.path.join(self.path, image)
            img_x = cv2.imread(img_path)

            # Update progress bar (0–30%)
            self.progress_reporter.report(value=(image_n / n_images) * 30)

            # Apply preprocessing (e.g., filtering, contrast)
            img_x = preprocess_init.preprocess(img_x)

            # Iterate over crop boxes (one per seedling)
            for n, box in enumerate(self.crop_boxes):
                center_x, center_y = box["cx"], box["cy"]
                x_1 = max(0, center_x - box["half_w"])
                y_1 = max(0, center_y - box["half_h"])
                x_2 = min(img_x.shape[1], center_x + box["half_w"])
                y_2 = min(img_x.shape[0], center_y + box["half_h"])

                crop = img_x[y_1:y_2, x_1:x_2]

                # Pad crop to 1024x1024 if near edges
                pad_y = self.crop_size - crop.shape[0]
                pad_x = self.crop_size - crop.shape[1]

                if pad_x > 0 or pad_y > 0:
                    crop = cv2.copyMakeBorder(
                        crop,
                        0, pad_y,
                        0, pad_x,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]  # Black padding
                    )
                
                crop = img_x[y_1:y_2, x_1:x_2]

                if torch.cuda.is_available():
                    #output of superres model is 4x the original image 
                    crop_x4=self.model_superres.enhance(crop)
                    self.crop_size = 1024
                    #downsize the image back to its original form
                    crop=cv2.resize(crop_x4, (self.crop_size,self.crop_size), interpolation = cv2.INTER_AREA)
                    #Superresolution of image

                cv2.imwrite(self.path_save_x+f'/{n}-crop-{image[:-4]}.png', crop)
        
        self.run_apical_pipeline()

        print(self.metadata_list)
        print(self.time_deltas)
        self.progress_reporter.report(done=True)


    def run_apical_pipeline(self):
        # Predict masks with Pytorch CNN from RootPainter
        cotyledon_predictor = UNetInference(model_path="weights/RootPainter_weights/cotyledon_v5.pkl")
        cotyledon_predictor3 = UNetInference(model_path="weights/RootPainter_weights/cotyledon_v3.pkl")
        hypocot_predictor = UNetInference(model_path="weights/RootPainter_weights/hypocot_v5.pkl")
        germ_predictor = UNetInference(model_path="weights/RootPainter_weights/germ_v1.pkl")

        cotyledon_predictor.predict_folder(image_dir="data/images/", output_dir="data/predict/", label="1")
        cotyledon_predictor3.predict_folder(image_dir="data/images/", output_dir="data/predict/", label="3")
        hypocot_predictor.predict_folder(image_dir="data/images/", output_dir="data/predict/", label="2")
        germ_predictor.predict_folder(image_dir="data/images/", output_dir="data/predict/", label="4")

        # Initialize angle data
        angle_df = pd.DataFrame(columns=["filename", "seedling_id", "angles", "tot_numb"])
        image_crops_angles = {}
        image_crops_angles_max = {}

        # Construct filenames
        self.cropped_sorted_filenames = [
            f"{crop_n}-crop-{filename[:-4]}.png"
            for crop_n in range(len(self.transformed_mid_points))
            for filename in self.file_list
        ]

        debug_data_batch = {}
        total_files = len(self.cropped_sorted_filenames)

        cotyl_time_series_contours = []
        germ_time_series_contours = []

        for idx, file_name in enumerate(self.cropped_sorted_filenames):
            self.progress_reporter.report(value=30 + int((idx / total_files) * 70))
            print(file_name)

            crop_id = int(file_name[0])
            image_path = os.path.join("data/images", file_name)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            cotyl_mask_path = f"data/predict/{file_name[:-4]}-1.png"
            hypo_mask_path = f"data/predict/{file_name[:-4]}-2.png"
            germ_mask_path = f"data/predict/{file_name[:-4]}-4.png"

            img_cotyl = cv2.imread(cotyl_mask_path, cv2.IMREAD_GRAYSCALE)
            img_hypo = cv2.imread(hypo_mask_path, cv2.IMREAD_GRAYSCALE)

            img_germ = cv2.imread(germ_mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or img_cotyl is None or img_hypo is None:
                print(f"[WARNING] Skipping {file_name}: image or masks not found.")

            # Process masks
            _, bin_cotyl = cv2.threshold(img_cotyl, 127, 255, cv2.THRESH_BINARY)
            _, bin_hypo = cv2.threshold(img_hypo, 127, 255, cv2.THRESH_BINARY)
            _, bin_germ = cv2.threshold(img_germ, 127, 255, cv2.THRESH_BINARY)

            seed_points = self.crop_points_distributed[crop_id]
            seed_ids = self.crop_points_numberID[crop_id]

            above_mask = cv2.bitwise_not(mask_below_seed_line(img_hypo.shape, seed_points))

            cotyl_mask = cv2.bitwise_and(cv2.bitwise_not(bin_cotyl), cv2.bitwise_not(bin_cotyl), mask=above_mask)
            hypo_mask = cv2.bitwise_and(cv2.bitwise_not(bin_hypo), cv2.bitwise_not(bin_hypo), mask=above_mask)
            
            # Define kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # Apply erosion and dilation
            cotyl_mask = cv2.erode(cotyl_mask, kernel, iterations=1)
            cotyl_mask = cv2.dilate(cotyl_mask, kernel, iterations=1)

            hypo_mask = cv2.erode(hypo_mask, kernel, iterations=1)
            hypo_mask = cv2.dilate(hypo_mask, kernel, iterations=1)

            cotyl_mask = PostprocessMasks.zoom_out_mask(cotyl_mask, scale=0.98)
            hypo_mask = PostprocessMasks.zoom_out_mask(hypo_mask, scale=0.98)
            germ_mask = PostprocessMasks.zoom_out_mask(bin_germ, scale=0.98)

            # Extract contours
            cotyl_contours = [c for c in cv2.findContours(cotyl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
            hypo_contours = [c for c in cv2.findContours(hypo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
            germ_contours = [c for c in cv2.findContours(germ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]

            cotyl_time_series_contours.append(cotyl_contours)
            germ_time_series_contours.append(germ_contours)

            print(f"Seed points: {seed_points}")
            print(f"Cotyl contours: {len(cotyl_contours)} | Hypo contours: {len(hypo_contours)}")

            # Apical hook angle computation
            hook = ApicalHook(
                img_name=file_name,
                time_series_contours = cotyl_time_series_contours,
                germ_time = germ_time_series_contours,
                hypo_mask=hypo_mask,
                seedling_points=seed_points,
                point_ids=seed_ids,
                image=image,
                )

            hook.process()
            hook.save(f"data/final_prediction/{file_name}")
            angle_dict = hook.get_angles()

            self.angle_handler = AngleDictHandler()
            # angle_dict = self.angle_handler._parse_angle_list(angles_raw)  # Uses internal method for cleaning
            print(f"Angles found for {seed_ids}: {angle_dict}")

            # Build old-style angle list aligned with seed_ids
            angle_list = []
            for sid in seed_ids:
                angle = angle_dict.get(sid, '-')
                if isinstance(angle, (int, float)) and not np.isnan(angle):
                    angle_list.append(round(angle))
                else:
                    angle_list.append('-')  # Or use np.nan

            # Save old-style format
            row = pd.DataFrame([[file_name, seed_ids, angle_list, seed_ids]],
                               columns=angle_df.columns)
            angle_df = pd.concat([angle_df, row], ignore_index=True)

            # Save max angles (this part stays the same)
            image_crops_angles[crop_id] = angle_dict
            max_dict = image_crops_angles_max.setdefault(crop_id, {})
            for sid, angle in angle_dict.items():
                if isinstance(angle, (int, float)) and not np.isnan(angle):
                    if sid not in max_dict or angle > max_dict.get(sid, float('-inf')):
                        max_dict[sid] = angle


         #    # Update max angles per seedling
         #    # Store angle dict in crop map
         #    image_crops_angles[crop_id] = angle_dict
# 
         #    # Step 8: Update max angles per seedling
         #    max_dict = image_crops_angles_max.setdefault(crop_id, {})
         #    for sid, angle in angle_dict.items():
         #        if isinstance(angle, (int, float)) and not np.isnan(angle):
         #            if sid not in max_dict or angle > max_dict.get(sid, float('-inf')):
         #                max_dict[sid] = angle
# 
         #    # Step 9: Save new row to angle_df
         #    row = pd.DataFrame([[file_name, seed_ids, str(angle_dict), seed_ids]],
         #                       columns=angle_df.columns)
         #    angle_df = pd.concat([angle_df, row], ignore_index=True)

        # Germination time-zero per seedling, from the germ_v1 mask time series.
        # germ_time_series_contours accumulates one entry per (crop_id, filename)
        # iteration above, crop_id-major (outer loop), so each seedling's own
        # frames form a contiguous num_frames-sized block.
        num_frames = len(self.file_list)
        self.germination_detector = GerminationDetector(crop_size=self.crop_size)
        for crop_id in range(len(self.transformed_mid_points)):
            frame_contours = germ_time_series_contours[crop_id * num_frames:(crop_id + 1) * num_frames]
            seed_point = self.crop_points_distributed[crop_id][0]
            self.germination_detector.detect(crop_id, frame_contours, seed_point)
        print(f"Germination time-zero per seedling: {self.germination_detector.germination_frame}")

        # Step 9: Save results
        angle_df.to_csv("img_angle_data.csv", index=False)
        self.img_angle_data = angle_df

        if self.debug_var.get() == 0:
            with open("data/json_data/json_data.json", "w") as f:
                json.dump(debug_data_batch, f)

        self.save_data()


    """
    Toggle "adjust crop boxes" mode: drag any seedling's corner handles to
    resize/reposition its auto-derived crop box before running the analysis.
    """
    def crop_image(self):
        if self.check_place_rect==False:
            self.button_crop.configure(bg="#d78a5e")
            self.check_place_rect=True
            self.root.bind("<Button-1>", self._start_drag_handle)
            self.root.bind("<B1-Motion>", self._drag_handle)
            self.root.bind("<ButtonRelease-1>", self._end_drag_handle)

        else:
            self.button_crop.configure(bg="white")
            self.check_place_rect=False
            self.root.unbind("<Button-1>")
            self.root.unbind("<B1-Motion>")
            self.root.unbind("<ButtonRelease-1>")
            self.button_start_analysis["state"]=tk.NORMAL
            self.button_preview_seedlings["state"]=tk.NORMAL



    #Converts the crop filename to the original filename
    def convert_filename(self, filename):
        filename_n=filename[7:-4]+self.img_format
        return(filename_n)
    
    def update_listbox_for_angles_miss(self, file_name):
        """Updates image label, angle data, and listbox display for a given filename using AngleDictHandler."""
        width, height = 200, 200
        self.listbox.delete(0, tk.END)

        # Update image label
        self.current_img_name.destroy()
        self.current_img_name = tk.Label(self.root, text=file_name[:-4])
        self.current_img_name.place(x=width + 500, y=height - 160)

        clean_filename = str(file_name).strip()

        if clean_filename not in self.angle_dataframe.index:
            print(f"[DEBUG] No angle data found for '{clean_filename}' — available keys: {list(self.angle_dataframe.index)}")
            return False
        
        angle_dataframe_x = self.angle_dataframe.loc[[clean_filename]]

        tot_numb = ast.literal_eval(angle_dataframe_x['tot_numb'].tolist()[0])

        # Get angles using handler
        self.angle_n = self.angle_handler.get_angles_for_image(file_name, tot_numb)

        print("[DEBUG] Angle list for display:")
        for sid, angle in self.angle_n:
            print(f"{sid}: {angle}")
            str1 = f'{sid:<10}{angle}°'
            self.listbox.insert(tk.END, str1)

        return True

    def update_listbox_for_angles(self, file_name):
        """Updates image label, angle data, and listbox display for a given filename."""
        width, height = 200, 200
        self.listbox.delete(0, tk.END)

        self.current_img_name.destroy()
        self.current_img_name = tk.Label(self.root, text=file_name[:-4])
        self.current_img_name.place(x=width + 500, y=height - 160)

        angle_dataframe_x = self.angle_dataframe.loc[self.angle_dataframe['filename'] == file_name]
        if angle_dataframe_x.empty:
            print(f"[DEBUG] No angle data found for {file_name}")
            return False  # signal: skip rendering image

        org_filename = self.convert_filename(file_name)
        tot_numb = ast.literal_eval(angle_dataframe_x['tot_numb'].tolist()[0])
        angle_data = self.new_df.loc[self.new_df['img_name'] == org_filename]
        angle_d = angle_data.squeeze()

        self.angle_n = []
        for seedling_id_n in tot_numb:
            try:
                angle = angle_d.get(int(seedling_id_n), np.nan)

                # If angle is a Series or array, squeeze it down
                if isinstance(angle, (np.ndarray, pd.Series)):
                    angle = float(np.squeeze(angle))

                if isinstance(angle, (int, float)) and not np.isnan(angle):
                    self.angle_n.append((seedling_id_n, int(angle)))
                else:
                    self.angle_n.append((seedling_id_n, '-'))

            except Exception as e:
                print(f"[DEBUG] Error with seedling {seedling_id_n}: {e}")
                self.angle_n.append((seedling_id_n, '-'))

            print(f"[DEBUG] Raw angle for seedling {seedling_id_n}: {angle_d.get(int(seedling_id_n))}")


        print("[DEBUG] Angle list for display:")
        for ang_n in self.angle_n:
            print(f"{ang_n[0]}: {ang_n[1]}")
            str1 = f'{ang_n[0]:<10}{ang_n[1]}°'
            self.listbox.insert(tk.END, str1)

        return True

    def next_angle_img(self):
        width, height = 200, 200
        if self.n_angle_image != self.checker_next_img_limit - 1:
            self.n_angle_image += 1
            file_name = self.images_final[self.n_angle_image]
            if not self.update_listbox_for_angles(file_name):
                return
            img_path = self.img_final_folder + file_name
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (600, 600))
            self.photo_n = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo_n, anchor=tk.NW)
        else:
            self.save_button['state'] = tk.NORMAL

    def previous_angle_img(self):
        width, height = 200, 200
        if self.n_angle_image != 0:
            self.n_angle_image -= 1
        if self.n_angle_image < self.checker_next_img_limit:
            file_name = self.images_final[self.n_angle_image]
            if not self.update_listbox_for_angles(file_name):
                return
            img_path = self.img_final_folder + file_name
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (600, 600))
            self.photo_n = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.delete(self.image_on_canvas)
            self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo_n, anchor=tk.NW)


    def openNewWindow(self):
        #remove buttons
        self.n_angle_image=0
        self.listbox.destroy()
        self.btn_import_image.destroy()
        self.logo_label.destroy()
        self.btn_show_image.destroy()
        self.button_crop.destroy()
        self.button_place_point.destroy()
        #self.btn_sort.destroy()
        #self.sort_txt_box.destroy()
        #self.sort_start_b.destroy()
        #self.sort_end_b.destroy()
        self.label_user_input.destroy()
        self.label_1.destroy()
        self.label_2.destroy()
        self.label_3.destroy()
        self.progress.destroy()
        self.button_start_analysis.destroy()
        self.check_button_debug.destroy()
        self.progress_bar_label.destroy()

        self.fin.destroy()

        width=200
        height=200
        self.fin = tk.Frame(self.root, width=200, height=200)
        self.fin.pack()
        self.fin.place(x=20, y=20)
        self.canvas = tk.Canvas(self.fin, bg='#FFFFFF', width=600, height=600)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


        label_rotate=tk.Label(self.root, text ='Cotyledon rotation')
        label_rotate.place(x=width + 465, y=height + 256)
        self.btn_blank_angle = tk.Button(self.root, text="Cotyledon in rotation", command=self.blank_angle)
        self.btn_blank_angle.place(x=width + 450, y=height + 281)

        label_show_image=tk.Label(self.root, text ='Use the arrows to navigate\n through image time points')
        label_show_image.place(x=width + 450, y=height + 320)
        

        self.buttonNext_angle_img = tk.Button(self.root, text="-->", width=4, command=self.next_angle_img)
        self.buttonNext_angle_img.place(x=width + 520, y=height + 355)


        self.buttonPrevious_angle_img = tk.Button(self.root, text="<--", width=4, command=self.previous_angle_img)
        self.buttonPrevious_angle_img.place(x=width + 480, y=height + 355)
        
        self.save_button = tk.Button(self.root, text="Save data", width=15, command=self.save_csv_data)
        self.save_button.place(x=width + 460, y=height + 400)
        self.save_button['state']=tk.DISABLED


        label_manual_ang=tk.Label(self.root, text ='Manual angle :')
        label_manual_ang.place(x=width + 460, y=height + 115)

        self.overhook_var=tk.IntVar()
        self.check_button_overhook= tk.Checkbutton(self.root, text='Overhook',variable=self.overhook_var)
        self.check_button_overhook.place(x=width + 460, y=height + 150)

        self.buttonPlace_angle = tk.Button(self.root, text="Place angle", width=15, command=self.activate_manual_ang)
        self.buttonPlace_angle.place(x=width + 460, y=height + 180)
        self.buttonReplace_angle = tk.Button(self.root, text="Replace angle", width=15, command=self.replace_angle)
        self.buttonReplace_angle.place(x=width + 460, y=height + 210)


        Label_img_name = tk.Label(self.root,text ='Image :')
        Label_img_name.place(x=width +450, y=height - 160)

        Label_seedling = tk.Label(self.root,text ='#Seedling')
        Label_seedling.place(x=width + 450, y=height - 120)
        Label_seedling_angle = tk.Label(self.root,text ='Angle')
        Label_seedling_angle.place(x=width + 515, y=height - 120)


        self.listbox = tk.Listbox(self.root, width=22, height=12)
        self.listbox.place(x=width + 450, y=height - 100)
        self.img_final_folder='data/final_prediction/'
        self.images_final=self.cropped_sorted_filenames


        self.checker_next_img_limit=len(self.images_final)
        self.angle_dataframe=pd.read_csv('img_angle_data.csv')

        first_image = self.images_final[0]

        # Display image name label
        self.current_img_name = tk.Label(self.root, text=first_image[:-4])
        self.current_img_name.place(x=width + 500, y=height - 160)

        # Update listbox and angle values using shared method
        success = self.update_listbox_for_angles(first_image)
        if not success:
            print(f"[DEBUG] No angle data found for {first_image}")
            return

        # Load and display image on canvas
        img_path = self.img_final_folder + first_image
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (600, 600))
        self.photo_n = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo_n, anchor=tk.NW)

        self.root.geometry("%dx%d+0+0" % (825, 650))
        # self.root.mainloop()

    def save_csv_data(self):
        file_formate=[('CSV-file','*.csv')]
        save_file_path = asksaveasfilename(filetypes= file_formate, defaultextension=file_formate)
        self.new_df.to_csv(save_file_path, index=False)
        """
        This function formats the data in the final form and save it to the path that the user gives.
        The data is saved with the filenames(timepoints) as the first  column, the rest of the columns represents
        each seedling-id-number 
        """

    def save_data(self):
        df = pd.read_csv('img_angle_data.csv')

        file_names = df['filename'].tolist()
        files = self.file_list
        crops = list(range(len(self.transformed_mid_points)))

        img_num = len(file_names) / len(crops)
        img_name_matrix = [['' for _ in crops] for _ in range(int(img_num))]

        crop_filenames = list(range(len(file_names)))
        for n in range(int(img_num)):
            filename_index = crop_filenames[n::int(img_num)]
            for i, index in enumerate(filename_index):
                img_name_matrix[n][i] = file_names[index]

        new_list = ast.literal_eval(df['tot_numb'].iloc[-1])
        last_seedling_id = new_list[-1]
        data = ['img_name'] + list(range(1, last_seedling_id + 1))

        self.new_df = pd.DataFrame(columns=data)

        main_ang, main_ids = [], []

        for n, element in enumerate(img_name_matrix):
            angle_data = [files[n]]
            seedling_ids_list = ['img_name']

            for name in element:
                df_n = df.loc[df['filename'] == name]
                if df_n.empty:
                    continue

                # Parse and clean angles
                angle_raw = df_n['angles'].tolist()[0]
                try:
                    angles_parsed = ast.literal_eval(angle_raw)
                except Exception:
                    angles_parsed = {}

                # Determine format and convert to dict
                if isinstance(angles_parsed, list):
                    seed_ids = ast.literal_eval(df_n['seedling_id'].tolist()[0])
                    angles_dict = {sid: val for sid, val in zip(seed_ids, angles_parsed)}
                else:
                    angles_dict = angles_parsed

                # Build cleaned angle list
                cleaned_angles = []
                for sid in ast.literal_eval(df_n['seedling_id'].tolist()[0]):
                    angle = angles_dict.get(sid, " ")
                    if isinstance(angle, (float, int)) and not np.isnan(angle):
                        cleaned_angles.append(round(angle))
                    else:
                        cleaned_angles.append(" ")

                angle_data += cleaned_angles
                seedling_ids_list += ast.literal_eval(df_n['seedling_id'].tolist()[0])

            main_ang.append(angle_data)
            main_ids.append(seedling_ids_list)

        for n in range(len(img_name_matrix)):
            row_dict = dict(zip(main_ids[n], main_ang[n]))
            row_df = pd.DataFrame([row_dict])
            self.new_df = pd.concat([self.new_df, row_df], ignore_index=True)


    def save_data_2(self):
        df=pd.read_csv('img_angle_data.csv')

        file_names=df['filename'].tolist()

        files=self.file_list
        crops=[n for n in range(len(self.transformed_mid_points))]

        img_num=(len(file_names)/len(crops))
        img_name_matrix=[['' for n in range(len(crops))] for i in range(int(img_num))]

        crop_filenames=[n for n in range(len(file_names))]
        for n in range(int(img_num)):
            filename_index=crop_filenames[n::int(img_num)]
            for i,index in enumerate(filename_index):
                img_name_matrix[n][i]=file_names[index]

        new_list=df['tot_numb'].iloc[-1]
        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)

        new_list=df['tot_numb'].iloc[-1]
        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)

        self.new_df=pd.DataFrame(columns=data)
        main_ang=[]
        main_ids=[]
        for n,element in enumerate(img_name_matrix):
            angle_data=[files[n]]
            seedling_ids_list=['img_name']
            for name in element:
                df_n=df.loc[df['filename']==name]
                angles=df_n['angles'].tolist()
                angles=angles[0]
                angles=ast.literal_eval(angles)
                angle_data=angle_data+angles
                
                seedling_ids=df_n['seedling_id'].tolist()
                seedling_ids=seedling_ids[0]
                seedling_ids=ast.literal_eval(seedling_ids)
                seedling_ids_list=seedling_ids_list+seedling_ids
            main_ang.append(angle_data)
            main_ids.append(seedling_ids_list)
        for n in range(len(img_name_matrix)):
            zip_iterator = zip(main_ids[n], main_ang[n])
            a_dictionary = dict(zip_iterator)
            row_df = pd.DataFrame([a_dictionary])
            self.new_df = pd.concat([self.new_df, row_df], ignore_index=True) # Pandas 2.0 removed method df.append(), use pd.concat()

    def save_data_archive(self):
        df=pd.read_csv('img_angle_data.csv')

        file_names=df['filename'].tolist()

        files=self.file_list
        crops=[n for n in range(len(self.transformed_mid_points))]

        img_num=(len(file_names)/len(crops))

        img_name_matrix=[['' for n in range(len(crops))] for i in range(int(img_num))]

        crop_filenames=[n for n in range(len(file_names))]
        for n in range(int(img_num)):
            filename_index=crop_filenames[n::int(img_num)]
            for i,index in enumerate(filename_index):
                img_name_matrix[n][i]=file_names[index]

        new_list=df['tot_numb'].iloc[-1]
        print(f"[INFO] seedling id {new_list}")

        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)



        new_list=df['tot_numb'].iloc[-1]
        new_list = ast.literal_eval(new_list)
        last_seedling_id=new_list[-1]
        data=['img_name']
        for n in range(1, last_seedling_id+1):
            data.append(n)


        self.new_df=pd.DataFrame(columns=data)
        main_ang=[]
        main_ids=[]
        for n,element in enumerate(img_name_matrix):
            angle_data=[files[n]]
            seedling_ids_list=['img_name']
            for name in element:
                df_n=df.loc[df['filename']==name]
                angles=df_n['angles'].tolist()
                angles=angles[0]
                angles=ast.literal_eval(angles)
                angle_data += list(angles.items())
                
                seedling_ids=df_n['seedling_id'].tolist()
                seedling_ids=seedling_ids[0]
                seedling_ids=ast.literal_eval(seedling_ids)
                seedling_ids_list=seedling_ids_list+seedling_ids
            main_ang.append(angle_data)
            main_ids.append(seedling_ids_list)
        for n in range(len(img_name_matrix)):
            zip_iterator = zip(main_ids[n], main_ang[n])
            a_dictionary = dict(zip_iterator)
            row_df = pd.DataFrame([a_dictionary])
            self.new_df = pd.concat([self.new_df, row_df], ignore_index=True) # Pandas 2.0 removed method df.append(), use pd.concat()
            #self.new_df = self.new_df.append(a_dictionary, ignore_index=True)


    def add_files(self, path_input=None):
        if path_input!=None:
            self.listbox.delete(0, tk.END)
            for file_n in path_input:
                self.listbox.insert(tk.END, file_n)
            self.file_list=path_input

        if self.filenames_listbox==True and path_input==None:
            self.listbox.delete(0, tk.END)
            self.path=askdirectory()
            folder=self.path.split('/')
            self.save_path=self.path_crop+folder[-1]

            self.file_list=sorted(os.listdir(self.path))

            path=sorted(os.listdir(self.path))


            for file_n in path:
                self.listbox.insert(tk.END, file_n)


        if self.filenames_listbox==False and path_input==None:
            #Activate buttons
            self.btn_show_image["state"]=tk.NORMAL
            #self.btn_sort["state"]=tk.NORMAL


            self.path=askdirectory()
            folder=self.path.split('/')
            self.save_path=self.path_crop+folder[-1]
            self.file_list=os.listdir(self.path)
            path=sorted(os.listdir(self.path))

            for file_n in path:
                self.listbox.insert(tk.END, file_n)

            self.filenames_listbox=True

            # Apply sorting logic
        sorted_filenames = []
        for file in self.file_list:
            base_name = os.path.splitext(file)[0]
            number_match = re.search(r'\d+', base_name)
            if number_match:
                sort_key = int(number_match.group())
                sorted_filenames.append((sort_key, file))
            else:
                print(f"[DEBUG] No number found in: {file}")
    
        sorted_filenames.sort(key=lambda x: x[0])
        self.file_list = [f[1] for f in sorted_filenames]
        
            # Select and show the last image
        if self.file_list:
            last_index = len(self.file_list) - 1
            self.listbox.select_set(last_index)
            self.listbox.event_generate("<<ListboxSelect>>")
            self.show_image()

        
    #Displays the image in the tkinter window 
    def show_image(self):
        self.button_place_point["state"]=tk.NORMAL
        # self.sort_start_b["state"]=tk.NORMAL
        # self.sort_end_b["state"]=tk.NORMAL

        clicked_file= self.listbox.curselection()
        for item in clicked_file:

            if self.show_image_check==True:
                
                #Image-file format
                self.img_format=self.listbox.get(item)[-4:]

                self.canvas.delete(self.image_on_canvas)

                self.image_n = cv2.imread(self.path+"/"+self.listbox.get(item))
                image_n=self.image_n
                # height1, width1, _ = image_n.shape
                self.height1, self.width1, _ = image_n.shape

                #Ratio of image axis after reshape
                self.Rx=(1100/((self.rectangle_width*2)-2))
                self.Ry=(650/((self.rectangle_height*2)-2))


                image_n=cv2.resize(image_n, (1100,650))
                self.image_n2=image_n

                self.photo_n = ImageTk.PhotoImage(image=Image.fromarray(image_n))
                self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo_n, anchor=tk.NW)
                self.root.mainloop()


            if self.show_image_check==False:

                self.show_image_check=True

                image_n = cv2.imread(self.path+"/"+self.listbox.get(item))
                self.img_format=self.listbox.get(item)[-4:]
                # height1, width1, channels1 = image_n.shape
                self.height1, self.width1, _ = image_n.shape
                
                # self.Rx=(1100/((self.rectangle_width*2)-2))
                # self.Ry=(650/((self.rectangle_height*2)-2))
                image_n=cv2.resize(image_n, (1100,650))

                self.photo_n = ImageTk.PhotoImage(image=Image.fromarray(image_n))
                self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo_n, anchor=tk.NW)
                self.root.mainloop()



#Remove image files at exit of software
def remove_files_exit():
    RemoveData()

atexit.register(remove_files_exit)

if __name__== '__main__':
    remove_files_exit()
    root=tk.Tk()
    gui=Gui(root)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (1280, 700))
    root.mainloop()