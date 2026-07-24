import os
import tkinter as tk
import numpy as np
import pandas as pd
import shutil
import atexit
import sys

# from subprocess import list2cmdline
from tkinter.ttk import Progressbar, Style, Separator
from tkinter.filedialog import asksaveasfilename, askdirectory
from tkinter import simpledialog
from models.UNetInference import *  # RootPainter
# from numpy.lib.function_base import select
from utils.apicalhook_angle import *
from utils.clean_on_exit import *
from utils.matching_crop2points_GUI import *
from utils.preprocess_model_input import *
from utils.postprocmask import PostprocessMasks, point_num, resolve_mask_path
from utils.gui_thread_safety import ProgressReporter
from utils.germination_detector import GerminationDetector
from ui.analysis_window import SeedlingAnalysisWindow
from ui.kinematics_window import KinematicsWindow

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

        # Per-seedling (crop_id) computed frame results/filenames, populated by
        # SeedlingAnalysisWindow (and by _export_results for any seedling never
        # previewed) -- kept on Gui rather than the Toplevel instance so a
        # seedling's results (including manual angle overrides) survive its
        # preview window being closed, and are visible to "Export results".
        self.frame_results_by_crop={}
        self.cropped_filenames_by_crop={}

        #The following variables are used to controll the buttons,
        self.check_place_rect=False
        self.crop_window_check=False
        self.rectangle_circle_match=[]
        self.point_numb=1
        self.button_circle_check=False
        self.show_image_check=False
        self.button_rect_check=False
        self.rectangle_width=0
        self.rectangle_height=0

        # Crops keep each seedling's own box-derived native pixel size --
        # never resized/padded to a fixed working size (UNetInference tiles
        # at native resolution, so there's no need for one).
        self.min_box_half_size = 30  # Degenerate-case floor only (e.g. start == end); boxes otherwise hug the start/end points + padding
        self.crop_padding_width_fraction = 0.4
        self.crop_padding_height_fraction = 0.10

        self.x_length=1100
        self.y_length=650
        width=self.x_length
        height=self.y_length
        self.sidebar_width=300

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

        # --- Canvas (left) ---
        self.canvas_frame = tk.Frame(self.root, width=width, height=height)
        self.canvas_frame.grid(row=0, column=0, sticky="nw")
        self.canvas_frame.grid_propagate(False)
        self.canvas = tk.Canvas(self.canvas_frame, bg='#FFFFFF', width=width, height=height)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # --- Sidebar (right) ---
        self.sidebar_frame = tk.Frame(self.root, width=self.sidebar_width)
        self.sidebar_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        sidebar_row = 0
        self.btn_import_image = tk.Button(self.sidebar_frame, text="Open image directory", width=17, command=self.add_files)
        self.btn_import_image.grid(row=sidebar_row, column=0, sticky="w", pady=(0, 6)); sidebar_row += 1

        self.listbox = tk.Listbox(self.sidebar_frame, width=25)
        self.listbox.grid(row=sidebar_row, column=0, sticky="w", pady=(0, 6)); sidebar_row += 1

        self.btn_show_image = tk.Button(self.sidebar_frame, text="Show image", width=10, command=self.show_image)
        self.btn_show_image["state"]=tk.DISABLED
        self.btn_show_image.grid(row=sidebar_row, column=0, sticky="w", pady=(0, 14)); sidebar_row += 1

        self.label_user_input = tk.Label(self.sidebar_frame, text = "User Input")
        self.label_user_input.grid(row=sidebar_row, column=0, sticky="w", pady=(0, 6)); sidebar_row += 1

        step1_frame = tk.Frame(self.sidebar_frame)
        step1_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_1 = tk.Label(step1_frame, text = "1.")
        self.label_1.pack(side=tk.LEFT)
        self.button_place_point=tk.Button(step1_frame, text="Place points", width=10, command=self.canvas_circle_activate)
        self.button_place_point.pack(side=tk.LEFT, padx=(6, 0))
        self.button_place_point["state"]=tk.DISABLED

        step2_frame = tk.Frame(self.sidebar_frame)
        step2_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_2 = tk.Label(step2_frame, text = "2.")
        self.label_2.pack(side=tk.LEFT)
        self.button_crop = tk.Button(step2_frame, text="Adjust crop", width=11, command=self.crop_image)
        self.button_crop.pack(side=tk.LEFT, padx=(6, 0))
        self.button_crop["state"]=tk.DISABLED

        step3_frame = tk.Frame(self.sidebar_frame)
        step3_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_3 = tk.Label(step3_frame, text = "3.")
        self.label_3.pack(side=tk.LEFT)
        self.button_start_analysis = tk.Button(step3_frame, text="Start Analysis", width=10, command=self._threading_analysis)
        self.button_start_analysis.pack(side=tk.LEFT, padx=(6, 0))
        self.button_start_analysis["state"]=tk.DISABLED

        step4_frame = tk.Frame(self.sidebar_frame)
        step4_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_4 = tk.Label(step4_frame, text = "4.")
        self.label_4.pack(side=tk.LEFT)
        self.button_preview_seedlings = tk.Button(step4_frame, text="Preview seedling", width=13, command=self.open_seedling_picker)
        self.button_preview_seedlings.pack(side=tk.LEFT, padx=(6, 0))
        self.button_preview_seedlings["state"]=tk.DISABLED

        step5_frame = tk.Frame(self.sidebar_frame)
        step5_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_5 = tk.Label(step5_frame, text = "5.")
        self.label_5.pack(side=tk.LEFT)
        self.button_kinematics = tk.Button(step5_frame, text="Show kinematics", width=13, command=self._open_kinematics_window)
        self.button_kinematics.pack(side=tk.LEFT, padx=(6, 0))
        self.button_kinematics["state"]=tk.DISABLED

        step6_frame = tk.Frame(self.sidebar_frame)
        step6_frame.grid(row=sidebar_row, column=0, sticky="w", pady=4); sidebar_row += 1
        self.label_6 = tk.Label(step6_frame, text = "6.")
        self.label_6.pack(side=tk.LEFT)
        self.button_export_results = tk.Button(step6_frame, text="Export results", width=13, command=self._export_results)
        self.button_export_results.pack(side=tk.LEFT, padx=(6, 0))
        self.button_export_results["state"]=tk.DISABLED

        sep = Separator(self.sidebar_frame, orient=tk.HORIZONTAL)
        sep.grid(row=sidebar_row, column=0, sticky="ew", pady=10); sidebar_row += 1

        self.debug_var=tk.IntVar(value=1)
        self.check_button_debug= tk.Checkbutton(self.sidebar_frame, text='Debug Mode', variable=self.debug_var)
        self.check_button_debug.grid(row=sidebar_row, column=0, sticky="w", pady=(0, 10)); sidebar_row += 1

        # --- Load and display the logo ---
        logo_image = Image.open("data/logo/logos.png")
        logo_image = logo_image.resize((160, 134), resample_filter)  # adjust size if needed
        self.logo_photo = ImageTk.PhotoImage(logo_image)  # keep a reference

        self.logo_label = tk.Label(self.sidebar_frame, image=self.logo_photo, borderwidth=0)
        self.logo_label.grid(row=sidebar_row, column=0, sticky="w"); sidebar_row += 1

        # --- Bottom bar (progress), spans full window width ---
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=8)
        self.progress_bar_label = tk.Label(self.bottom_frame, text = "Measurement progress:")
        self.progress_bar_label.pack(side=tk.LEFT)
        self.progress = Progressbar(self.bottom_frame, orient=tk.HORIZONTAL, length=700)
        self.progress.pack(side=tk.LEFT, padx=(10, 0))

        # Size the window to fit the actual content instead of a guessed size,
        # so it stays correct regardless of sidebar/logo width tweaks.
        self.root.update_idletasks()
        self.root.geometry(f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}")
        self.root.resizable(False, False)

        # Thread-safe channel for background analysis to report progress back to the GUI
        self.progress_reporter = ProgressReporter()
        self.root.after(100, self._pump_progress)


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

    def _compute_crop_box(self, start, end):
        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        half_w = (max_x - min_x) * (1 + 2 * self.crop_padding_width_fraction) / 2
        half_h = (max_y - min_y) * (1 + 2 * self.crop_padding_height_fraction) / 2

        # Only a degenerate-case floor (e.g. start == end) -- the box hugs
        # the start/end points plus padding, it is not floored up to any
        # fixed default size.
        half_w = max(half_w, self.min_box_half_size)
        half_h = max(half_h, self.min_box_half_size)

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

        half_w = max(abs(x2 - x1) / 2, self.min_box_half_size)
        half_h = max(abs(y2 - y1) / 2, self.min_box_half_size)

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
                self.button_crop["state"] = tk.NORMAL
                self.button_start_analysis["state"] = tk.NORMAL
                self.button_place_point["state"] = tk.NORMAL
                self.btn_show_image["state"] = tk.NORMAL
                self.btn_import_image["state"] = tk.NORMAL
                self.progress_bar_label.configure(
                    text="Analysis complete -- use Preview seedling to review, or Export results when done")
            if event.get("export_done"):
                self.button_export_results["state"] = tk.NORMAL
                file_formats = [('CSV-file', '*.csv')]
                save_file_path = asksaveasfilename(filetypes=file_formats, defaultextension=file_formats)
                if save_file_path:
                    event["export_df"].to_csv(save_file_path, index=False)
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

                if torch.cuda.is_available():
                    # Super-resolution enhancement (4x), then downsized back
                    # to this crop's own native size -- sharpens detail
                    # without altering the crop's actual pixel dimensions.
                    # UNetInference tiles at native resolution (no fixed
                    # working size), so the saved crop must preserve its own
                    # box-derived size, not be forced to any fixed square.
                    orig_h, orig_w = crop.shape[:2]
                    crop_x4 = self.model_superres.enhance(crop)
                    crop = cv2.resize(crop_x4, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

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
        angle_df = pd.DataFrame(columns=["filename", "seedling_id", "angles", "tot_numb", "states"])
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

        self.reset_frame_accumulators()

        for idx, file_name in enumerate(self.cropped_sorted_filenames):
            self.progress_reporter.report(value=30 + int((idx / total_files) * 70))
            print(file_name)

            crop_id = int(file_name[0])
            result = self.process_single_frame(file_name, crop_id)
            if result is None:
                continue

            hook = result["hook"]
            seed_ids = result["seed_ids"]
            angle_list = result["angle_list"]
            angle_dict = result["angle_dict"]
            state_list = result["state_list"]

            hook.save(f"data/final_prediction/{file_name}")
            print(f"Angles found for {seed_ids}: {angle_dict}")

            # Mirrors segment_single_seedling/SeedlingAnalysisWindow's own
            # bookkeeping, so a seedling already segmented by this batch pass
            # isn't redundantly re-segmented by "Preview seedling"/"Export results".
            self.frame_results_by_crop.setdefault(crop_id, []).append(result)
            self.cropped_filenames_by_crop.setdefault(crop_id, []).append(file_name)

            # Save old-style format
            row = pd.DataFrame([[file_name, seed_ids, angle_list, seed_ids, state_list]],
                               columns=angle_df.columns)
            angle_df = pd.concat([angle_df, row], ignore_index=True)

            # Save max angles (this part stays the same)
            image_crops_angles[crop_id] = angle_dict
            max_dict = image_crops_angles_max.setdefault(crop_id, {})
            for sid, angle in angle_dict.items():
                if isinstance(angle, (int, float)) and not np.isnan(angle):
                    if sid not in max_dict or angle > max_dict.get(sid, float('-inf')):
                        max_dict[sid] = angle

        # Germination time-zero per seedling, from the germ_v1 mask time series
        # process_single_frame accumulated above, keyed by crop_id.
        for crop_id in range(len(self.transformed_mid_points)):
            self._ensure_germination_detected(crop_id)
        print(f"Germination time-zero per seedling: {self.germination_detector.germination_frame}")

        # Step 9: Save results
        angle_df.to_csv("img_angle_data.csv", index=False)
        self.img_angle_data = angle_df

        if self.debug_var.get() == 0:
            with open("data/json_data/json_data.json", "w") as f:
                json.dump(debug_data_batch, f)

        self.save_data()

    def reset_frame_accumulators(self, crop_id=None):
        """Clears the per-crop_id contour history process_single_frame builds
        up as it walks a seedling's frames in order (ApicalHook/GerminationDetector
        need it in time order, so a fresh run mustn't append onto a stale one)."""
        if crop_id is None:
            self.cotyl_time_series_by_crop = {}
            self.germ_time_series_by_crop = {}
        else:
            self.cotyl_time_series_by_crop[crop_id] = []
            self.germ_time_series_by_crop[crop_id] = []

    def _ensure_germination_detected(self, crop_id):
        """Runs Phase 4's germination time-zero detection for one seedling,
        against whatever germ contour history process_single_frame has
        accumulated for it so far (self.germ_time_series_by_crop[crop_id]).
        Callable from any of the three places a seedling's frames get
        processed -- the "Start Analysis" batch, "Preview seedling", and
        "Export results" -- so germination time-zero isn't only ever known
        for seedlings that went through a full batch run. Safe to call more
        than once for the same seedling (e.g. previewed with fewer frames,
        then later re-segmented with more via Export results): each call
        just re-detects against the current, possibly fuller, history."""
        if self.germination_detector is None:
            self.germination_detector = GerminationDetector()
        frame_contours = self.germ_time_series_by_crop.get(crop_id, [])
        seed_point = self.crop_points_distributed[crop_id][0]
        box = self.crop_boxes[crop_id]
        # This seedling's own crop's pixel size -- crops keep their native,
        # box-derived dimensions (not resized to a shared fixed working
        # size), so the proximity radius/area threshold must scale per
        # seedling instead of assuming one global crop size.
        crop_size = (2 * box["half_w"] + 2 * box["half_h"]) / 2
        self.germination_detector.detect(crop_id, frame_contours, seed_point, crop_size)

    def _ensure_crop_points(self, crop_id):
        """Lazily computes crop_points_distributed[crop_id]/crop_points_numberID[crop_id]
        from just that seedling's own box + start point, so the on-demand preview
        path (segment_single_seedling/process_single_frame) doesn't depend on the
        full "Start Analysis" batch (start_analysis's MatchCropPoints/point_num)
        having run first. Valid because crop boxes are 1:1 with seedling start
        points by construction (Phase 2) -- this is the same math MatchCropPoints
        does per box, with no cross-seedling dependency.

        No rescaling is applied: the saved crop keeps this box's own native
        pixel dimensions (it is not resized to any fixed working size), so
        the point's position relative to the box's own origin is exactly its
        position in the saved crop file."""
        while len(self.crop_points_distributed) <= crop_id:
            self.crop_points_distributed.append(None)
            self.crop_points_numberID.append(None)
        if self.crop_points_distributed[crop_id] is not None:
            return

        box = self.crop_boxes[crop_id]
        px, py = self.seedling_pairs[crop_id]["start"]
        x1, y1 = box["cx"] - box["half_w"], box["cy"] - box["half_h"]
        self.crop_points_distributed[crop_id] = [(int(round(px - x1)), int(round(py - y1)))]
        self.crop_points_numberID[crop_id] = [crop_id + 1]

    def process_single_frame(self, file_name, crop_id, frame_index=None):
        """
        Mask load -> threshold -> erode/dilate -> contour -> ApicalHook.process()
        for one already-cropped, already-segmented frame. Shared by the full
        batch pipeline (run_apical_pipeline) and the Phase 5 per-seedling
        preview window, so both compute angles identically.

        Masks are loaded via resolve_mask_path(), which prefers a Phase 7
        brush-edited mask in data/postprocess/ over the raw data/predict/
        prediction when one exists -- this is the only hook Phase 7's
        "recompute after a manual mask edit" needs; the rest of this method
        (threshold/erode/dilate/contour) is unchanged either way.

        By default (frame_index=None) this appends onto
        self.cotyl_time_series_by_crop[crop_id]/germ_time_series_by_crop[crop_id],
        for callers walking a seedling's frames in order from the start
        (reset_frame_accumulators(crop_id) first for a fresh run). Passing
        frame_index instead REPLACES that one entry in place -- used by
        Phase 7's "Recompute angle" button, which reprocesses a single
        already-processed frame after an edit without disturbing the frames
        around it or duplicating history entries.

        Returns None if the image or a required mask is missing, else a dict
        with the ApicalHook instance, angle results, and the frame's masks
        (for a caller that wants to render an overlay preview).
        """
        self._ensure_crop_points(crop_id)

        image_path = os.path.join("data/images", file_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img_cotyl = cv2.imread(resolve_mask_path(file_name, "1"), cv2.IMREAD_GRAYSCALE)
        img_hypo = cv2.imread(resolve_mask_path(file_name, "2"), cv2.IMREAD_GRAYSCALE)
        img_germ = cv2.imread(resolve_mask_path(file_name, "4"), cv2.IMREAD_GRAYSCALE)

        if image is None or img_cotyl is None or img_hypo is None:
            print(f"[WARNING] Skipping {file_name}: image or masks not found.")
            return None

        # Process masks
        _, bin_cotyl = cv2.threshold(img_cotyl, 127, 255, cv2.THRESH_BINARY)
        _, bin_hypo = cv2.threshold(img_hypo, 127, 255, cv2.THRESH_BINARY)
        bin_germ = cv2.threshold(img_germ, 127, 255, cv2.THRESH_BINARY)[1] if img_germ is not None else None

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
        germ_mask = PostprocessMasks.zoom_out_mask(bin_germ, scale=0.98) if bin_germ is not None else None

        # Extract contours
        cotyl_contours = [c for c in cv2.findContours(cotyl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
        hypo_contours = [c for c in cv2.findContours(hypo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
        germ_contours = ([c for c in cv2.findContours(germ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
                          if germ_mask is not None else [])

        cotyl_history = self.cotyl_time_series_by_crop.setdefault(crop_id, [])
        germ_history = self.germ_time_series_by_crop.setdefault(crop_id, [])

        if frame_index is None:
            cotyl_history.append(cotyl_contours)
            germ_history.append(germ_contours)
            # ApicalHook only ever reads time_series_contours[-1] (see
            # RegionMatcher), so the full running history is fine here.
            cotyl_history_for_hook = cotyl_history
            germ_history_for_hook = germ_history
        else:
            cotyl_history[frame_index] = cotyl_contours
            germ_history[frame_index] = germ_contours
            # Truncate to end at this frame, so ApicalHook's [-1] lookup sees
            # the just-recomputed contours instead of whatever came after it.
            cotyl_history_for_hook = cotyl_history[:frame_index + 1]
            germ_history_for_hook = germ_history[:frame_index + 1]

        print(f"Seed points: {seed_points}")
        print(f"Cotyl contours: {len(cotyl_contours)} | Hypo contours: {len(hypo_contours)}")

        # Apical hook angle computation
        hook = ApicalHook(
            img_name=file_name,
            time_series_contours=cotyl_history_for_hook,
            germ_time=germ_history_for_hook,
            hypo_mask=hypo_mask,
            seedling_points=seed_points,
            point_ids=seed_ids,
            image=image,
            )

        hook.process()
        angle_dict = hook.get_angles()
        state_dict = hook.get_states()

        # Build old-style angle list aligned with seed_ids
        angle_list = []
        for sid in seed_ids:
            angle = angle_dict.get(sid, '-')
            if isinstance(angle, (int, float)) and not np.isnan(angle):
                angle_list.append(round(angle))
            else:
                angle_list.append('-')  # Or use np.nan

        # Bio state (Open/Opening/Closed/Overhooked) per seedling, aligned with
        # seed_ids the same way angle_list is -- Phase 8's kinematics graph
        # color-codes points by this.
        state_list = [state_dict.get(sid, '-') for sid in seed_ids]

        return {
            "hook": hook,
            "angle_dict": angle_dict,
            "angle_list": angle_list,
            "state_dict": state_dict,
            "state_list": state_list,
            "seed_ids": seed_ids,
            "cotyl_mask": cotyl_mask,
            "hypo_mask": hypo_mask,
            "germ_mask": germ_mask,
        }

    def crop_single_seedling(self, crop_id):
        """
        Crops just this one seedling's time series (same naming convention as
        the batch crop loop in start_analysis: "{crop_id}-crop-{filename}.png"),
        so the Phase 5 preview window can inspect one seedling before the full
        "Start Analysis" batch has run.
        """
        preprocess_init = preprocess_images()
        box = self.crop_boxes[crop_id]
        cropped_filenames = []

        for image in self.file_list:
            img_path = os.path.join(self.path, image)
            img_x = cv2.imread(img_path)
            img_x = preprocess_init.preprocess(img_x)

            center_x, center_y = box["cx"], box["cy"]
            x_1 = max(0, center_x - box["half_w"])
            y_1 = max(0, center_y - box["half_h"])
            x_2 = min(img_x.shape[1], center_x + box["half_w"])
            y_2 = min(img_x.shape[0], center_y + box["half_h"])

            crop = img_x[y_1:y_2, x_1:x_2]

            out_name = f"{crop_id}-crop-{image[:-4]}.png"
            cv2.imwrite(os.path.join(self.path_save_x, out_name), crop)
            cropped_filenames.append(out_name)

        return cropped_filenames

    def segment_single_seedling(self, crop_id, progress_reporter=None):
        """
        Crops (if needed) and runs UNetInference scoped to just this one
        seedling's own cropped frames, writing masks into data/predict/ same
        as the batch pipeline -- lets a Phase 5 preview window spot-check one
        seedling without running inference over every other seedling too.
        Runs on a background thread; progress_reporter must be a
        ProgressReporter the caller polls on the main thread.
        """
        reporter = progress_reporter or self.progress_reporter

        reporter.report(message=f"Cropping seedling {crop_id}...")
        self.reset_frame_accumulators(crop_id)
        cropped_filenames = self.crop_single_seedling(crop_id)
        file_paths = [os.path.join("data/images", f) for f in cropped_filenames]

        # cotyledon_v3's output ("-3.png") isn't read by process_single_frame
        # in the batch pipeline either -- skipped here to keep the on-demand
        # preview from running a model whose result nothing consumes.
        reporter.report(message="Running segmentation models...")
        cotyledon_predictor = UNetInference(model_path="weights/RootPainter_weights/cotyledon_v5.pkl")
        hypocot_predictor = UNetInference(model_path="weights/RootPainter_weights/hypocot_v5.pkl")
        germ_predictor = UNetInference(model_path="weights/RootPainter_weights/germ_v1.pkl")

        cotyledon_predictor.predict_files(file_paths, output_dir="data/predict/", label="1")
        hypocot_predictor.predict_files(file_paths, output_dir="data/predict/", label="2")
        germ_predictor.predict_files(file_paths, output_dir="data/predict/", label="4")

        reporter.report(message="Computing angles...")
        return cropped_filenames

    def open_seedling_picker(self):
        """Phase 5: lightweight Toplevel listing each seedling; opens a
        SeedlingAnalysisWindow for whichever one the user picks, on demand,
        rather than spawning a window per seedling all at once."""
        if not self.crop_boxes:
            return

        picker = tk.Toplevel(self.root)
        picker.title("Preview a seedling")

        listbox = tk.Listbox(picker, width=20, height=min(10, len(self.crop_boxes)))
        for i in range(len(self.crop_boxes)):
            listbox.insert(tk.END, f"Seedling {i}")
        listbox.pack(padx=8, pady=8)

        def _open_selected():
            selection = listbox.curselection()
            if not selection:
                return
            SeedlingAnalysisWindow(self, selection[0])

        listbox.bind("<Double-Button-1>", lambda event: _open_selected())
        tk.Button(picker, text="Open", command=_open_selected).pack(pady=(0, 8))

    def _export_results(self):
        """Replaces the removed legacy window's "Save data": consolidates
        every seedling's current angle data -- including manual overrides and
        mask-edit recomputes made via SeedlingAnalysisWindow -- into one
        wide-format (image x seedling) CSV. Runs on a background thread since
        it may need to segment any seedling never previewed."""
        self.button_export_results["state"] = tk.DISABLED
        threading.Thread(target=self._run_export_results).start()

    def _run_export_results(self):
        n_crops = len(self.crop_boxes)
        for crop_id in range(n_crops):
            if crop_id in self.frame_results_by_crop:
                continue
            self.progress_reporter.report(message=f"Segmenting seedling {crop_id}...",
                                           value=int((crop_id / n_crops) * 80))
            cropped_filenames = self.segment_single_seedling(crop_id, progress_reporter=self.progress_reporter)
            results = [self.process_single_frame(file_name, crop_id) for file_name in cropped_filenames]
            self.frame_results_by_crop[crop_id] = results
            self.cropped_filenames_by_crop[crop_id] = cropped_filenames

        self.progress_reporter.report(message="Detecting germination...", value=85)
        for crop_id in range(n_crops):
            self._ensure_germination_detected(crop_id)

        self.progress_reporter.report(message="Building CSV...", value=90)
        columns = ['img_name'] + list(range(1, n_crops + 1))
        rows = []
        for frame_idx, filename in enumerate(self.file_list):
            row = {'img_name': filename}
            for crop_id in range(n_crops):
                seed_id = crop_id + 1
                frame_results = self.frame_results_by_crop.get(crop_id, [])
                result = frame_results[frame_idx] if frame_idx < len(frame_results) else None
                angle = result["angle_dict"].get(seed_id) if result else None
                row[seed_id] = round(angle) if isinstance(angle, (int, float)) and not np.isnan(angle) else ''
            rows.append(row)
        export_df = pd.DataFrame(rows, columns=columns)

        self.progress_reporter.report(value=100, message="Export ready", export_df=export_df, export_done=True)

    def _open_kinematics_window(self):
        """Phase 8: opens the aggregate kinematics graph, reading whatever is
        currently in frame_results_by_crop -- works from Start Analysis,
        Preview seedling, or Export results alike."""
        KinematicsWindow(self)

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
            self.button_export_results["state"]=tk.NORMAL
            self.button_kinematics["state"]=tk.NORMAL



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
    root.mainloop()