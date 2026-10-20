import logging
import os
# Quiet OpenCV's libtiff warnings (e.g. 'Unknown field with tag 50839' from ImageJ/Adobe TIFFs): harmless
# but printed once per frame. Must be set BEFORE cv2 is first imported, here or via any module below.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import tkinter as tk
import numpy as np
import atexit
import sys

from tkinter.ttk import Progressbar, Style
from tkinter.filedialog import askdirectory
from tkinter import simpledialog, messagebox
from models.UNetInference import IMAGE_CHUNK
from models.segmentation_backends import (
    get_backend, get_germination_runner, get_landmark_runner, resolve_angle_source,
    resolve_backend_name, resolve_germination_method, HYPOCOTYL_BINARY_LABEL,
)
from utils.apicalhook_angle import ApicalHook
from utils.angle_timeseries import (reconstruct_series, sync_angle_and_state_lists, classify_states,
                                    UNREADABLE_STATE)
from utils.landmark_gate import gate_enabled
from utils.landmark_timeseries import reconstruct_landmark_series_detail
from utils.landmark_temporal import temporal_flags
from utils.hypocotyl_traits import hypocotyl_traits
from utils.clean_on_exit import RemoveData
from utils.preprocess_model_input import compute_time_deltas, metadata_has_gaps, order_series, preprocess_images
from utils.postprocmask import zoom_out_mask
from utils.mask_store import MaskStore
from utils.gui_thread_safety import ProgressReporter
from utils.germination_detector import GerminationDetector
from utils.crop_layout import find_overlapping_pairs
from ui.crop_box_canvas import CropBoxCanvas
from ui.gui import CropBoxMixin, ExportMixin, ScaleMixin, SeedlingWindowsMixin

import threading
import traceback

from PIL import Image, ImageTk

try:
    resample_filter = Image.Resampling.LANCZOS  # Pillow ≥ 10
except AttributeError:
    resample_filter = Image.ANTIALIAS  # Pillow < 10

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # in case cv2 was imported earlier
except AttributeError:
    pass

from torchvision.transforms import functional

sys.modules["torchvision.transforms.functional_tensor"] = functional

from utils.superres_flag import superres_requested
from utils.device import pick_device
from utils.paths import crop_path, data_path, images_dir, repo_path

log = logging.getLogger(__name__)


"""
This is the graphical user interface for the DLhook software

"""

# Extensions accepted when browsing an image folder (matched case-insensitively). Output names
# use os.path.splitext, so any extension length works.
VALID_IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


def _stem(file_name):
    """File name without its extension (any length: ".tif", ".tiff", ".jpeg", ...)."""
    return os.path.splitext(file_name)[0]


def _list_image_files(path):
    """Return sorted image filenames in `path`, skipping subfolders and
    non-image files (e.g. exported CSVs, Thumbs.db) that would otherwise be
    picked up by a plain os.listdir and later fail to load in cv2.imread."""
    return sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(path, f))
    )


class Gui(CropBoxMixin, ScaleMixin, ExportMixin, SeedlingWindowsMixin):
    def __init__(self, root):
        self.root = root
        self._init_superres()
        self._init_state()
        self._build_window()
        self._build_canvas()
        self._build_sidebar()
        self._build_status_bar()
        self._update_button_states()
        self._fit_window()

        # Thread-safe channel for background analysis to report progress back to the GUI
        self.progress_reporter = ProgressReporter()
        self.root.after(100, self._pump_progress)

    def _init_superres(self):
        """Optional Real-ESRGAN model (None unless DLHOOK_SUPERRES=1 and CUDA works)."""
        # Real-ESRGAN is opt-in (DLHOOK_SUPERRES=1, see utils/superres_flag.py) and only imported then
        self.model_superres = None
        if superres_requested() and pick_device().type == "cuda":
            from models.superres.superresolution_predict import RealesrganSuperresolution
            self.model_superres = RealesrganSuperresolution()
        sys.setrecursionlimit(10000)

    def _init_state(self):
        """Session state: images, crop boxes, per-seedling results and stores."""
        self.path_crop = images_dir()
        self.path_save_x = images_dir()


        self.angles=[]

        self.img_count=0
        self.photo_list=[]
        self.metadata_list=[]
        self.fallback_interval_minutes=None
        self.time_deltas=[]

        # Selected image folder, its frames in acquisition order, and each
        # frame's capture time (filename -> datetime or None), read once at
        # import by order_series and reused by _collect_image_metadata.
        self.path=None
        self.file_list=[]
        self.capture_times={}

        # Per-seedling crop boxes (image coords) {"cx","cy","half_w","half_h"}, drawn on the main
        # canvas (ui/crop_box_canvas.py) and kept clamped to the image and in spatial
        # (top-to-bottom, left-to-right) order by _on_boxes_changed, so list index == crop_id ==
        # label - 1. collar_overrides is parallel: the user's collar for that box, (x, y) in IMAGE
        # pixels, or None -- see _seed_point.
        self.crop_boxes=[]
        self.collar_overrides=[]
        # Headless callers only (multi/validate_angles.py): {crop_id: (x, y)} crop-coordinate seed
        # points used when neither a collar override nor a landmark collar exists.
        self.seed_point_proxies={}
        # Image size of the frame shown on the canvas (boxes are clamped to it).
        self.width1=0
        self.height1=0
        self.series_name=""
        self.current_frame_name=""
        self._busy=False
        self._exporting=False
        self._analysis_windows=[]

        self.transformed_mid_points=[]

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

        # In-memory segmentation masks, replacing the data/predict/ +
        # data/postprocess/ PNG round-trip. Lives on Gui (not on a preview
        # Toplevel) so masks survive a window being closed and reopened. Set DLHOOK_DUMP_MASKS=1 to also
        # write masks out to data/predict/ at the end of a run for validation
        # against previously exported ground-truth masks.
        self.mask_store = MaskStore()
        # Landmark-source readouts (utils/landmark_readout.py), keyed by crop filename
        # ("{crop_id}-crop-{name}.png"); value None = the model found no hook in that crop.
        # Filled by _segment_paths when the angle source is "landmarks".
        self.landmark_store = {}
        # Physical scale in micrometres per image pixel (None = not set; the export then has no mm
        # columns). Set from the "Scale" entry or the calibration window; kept on Gui for the
        # session (the app has no session save/load).
        self.scale_um_per_px = None
        # Folder for "Save edits for retraining" records (asked once by the analysis window) and the
        # path of the last exported CSV (the dialog's suggested location).
        self.landmark_edits_dir = None
        self.last_export_path = None
        self.dump_masks = os.environ.get("DLHOOK_DUMP_MASKS", "").strip() not in ("", "0", "false", "False")

        self.sidebar_width=280
        self.filenames_listbox=False

    def _build_window(self):
        """Window title, icon, style and the grid weights."""
        # configure style
        self.style = Style(self.root)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        icon_img = ImageTk.PhotoImage(file=repo_path("data/logo/icon.png"))
        self.icon_img = icon_img
        self.root.iconphoto(False, self.icon_img)
        # self.root.iconbitmap("@data/logo/icon.xbm")
        self.root.title('DLhook')

        # The canvas column grows with the window; the sidebar keeps its width.
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def _build_canvas(self):
        """The zoomable crop-box canvas (left)."""
        # --- Canvas (left): zoomable, draw / move / resize crop boxes ---
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas = CropBoxCanvas(
            self.canvas_frame, width=1000, height=650, on_change=self._on_boxes_changed,
            allow_collar=True, highlight_overlaps=True, show_labels=True, auto_resize=True,
            highlightthickness=0, bd=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind_view_keys()

    def _build_sidebar(self):
        """The sidebar (right): logos and the grouped controls."""
        # --- Sidebar (right): grouped controls ---
        self.sidebar_frame = tk.Frame(self.root, width=self.sidebar_width)
        self.sidebar_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=(10, 0))

        # --- Logos at the bottom of the sidebar ---
        # Packed BEFORE the groups: pack hands out space in packing order, so when the window is
        # short the last-packed widget is the one squeezed out -- the logos must not be it.
        logo_image = Image.open(repo_path("data/logo/logos.png"))
        logo_image = logo_image.resize((160, 134), resample_filter)
        self.logo_photo = ImageTk.PhotoImage(logo_image)  # keep a reference
        self.logo_label = tk.Label(self.sidebar_frame, image=self.logo_photo, borderwidth=0)
        self.logo_label.pack(side=tk.BOTTOM, pady=(4, 6))


        images_group = self._sidebar_group("Images")
        self.btn_import_image = tk.Button(images_group, text="Open image directory", command=self.add_files)
        self.btn_import_image.pack(fill=tk.X, pady=(0, 6))
        list_frame = tk.Frame(images_group)
        list_frame.pack(fill=tk.X)
        # exportselection=False keeps the selected frame highlighted while focus is elsewhere
        self.listbox = tk.Listbox(list_frame, width=30, height=6, exportselection=False)
        list_scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_frame_selected)

        seedlings_group = self._sidebar_group("Seedlings")
        seedlings_group.grid_columnconfigure((0, 1), weight=1, uniform="seedling_btns")
        self.btn_save_boxes = tk.Button(seedlings_group, text="Save boxes…", command=self.save_boxes)
        self.btn_save_boxes.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.btn_load_boxes = tk.Button(seedlings_group, text="Load boxes…", command=self.load_boxes)
        self.btn_load_boxes.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.btn_delete_box = tk.Button(seedlings_group, text="Delete selected",
                                        command=lambda: self.canvas.delete_selected())
        self.btn_delete_box.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.btn_clear_collar = tk.Button(seedlings_group, text="Clear collar",
                                          command=lambda: self.canvas.clear_collar())
        self.btn_clear_collar.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.btn_reset_zoom = tk.Button(seedlings_group, text="Reset zoom", command=lambda: self.canvas.reset_zoom())
        self.btn_reset_zoom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

        analysis_group = self._sidebar_group("Analysis")
        self.button_start_analysis = tk.Button(analysis_group, text="Start Analysis", command=self._threading_analysis)
        self.button_start_analysis.pack(fill=tk.X, pady=2)
        self.button_preview_seedlings = tk.Button(analysis_group, text="Preview seedling", command=self.open_seedling_picker)
        self.button_preview_seedlings.pack(fill=tk.X, pady=2)
        self.button_kinematics = tk.Button(analysis_group, text="Show kinematics", command=self._open_kinematics_window)
        self.button_kinematics.pack(fill=tk.X, pady=2)
        self.button_export_results = tk.Button(analysis_group, text="Export results", command=self._export_results)
        self.button_export_results.pack(fill=tk.X, pady=2)

        settings_group = self._sidebar_group("Settings")
        scale_frame = tk.Frame(settings_group)
        scale_frame.pack(fill=tk.X)
        tk.Label(scale_frame, text="Scale (µm/px)").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="")
        self.scale_entry = tk.Entry(scale_frame, width=8, textvariable=self.scale_var)
        self.scale_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.scale_entry.bind("<Return>", lambda event: self._apply_scale_entry(show_error=True))
        self.scale_entry.bind("<FocusOut>", lambda event: self._apply_scale_entry(show_error=False))
        self.btn_calibrate_scale = tk.Button(settings_group, text="Calibrate scale…", command=self._open_scale_calibration)
        self.btn_calibrate_scale.pack(fill=tk.X, pady=(4, 2))
        self.debug_var=tk.IntVar(value=0)
        self.check_button_debug= tk.Checkbutton(settings_group, text='Debug Mode', variable=self.debug_var)
        self.check_button_debug.pack(anchor="w")

    def _sidebar_group(self, title):
        frame = tk.LabelFrame(self.sidebar_frame, text=title, padx=8, pady=6)
        frame.pack(fill=tk.X, pady=(0, 8))
        return frame

    def _build_status_bar(self):
        """Status line, key help and the progress bar, spanning the window width."""
        # --- Status line, key help, and the progress bar (span the full window width) ---
        self.status_var = tk.StringVar(value="Open an image directory to start.")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w", font=("Helvetica", 10, "bold"))
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 0))
        help_text = ("Draw a seedling: drag on empty space   Move: drag a box   Resize: drag a corner of the selected box   "
                     "Select: click   Delete: Del / Backspace   Deselect: Esc   "
                     "Collar (optional): Shift+click inside a box, Shift+right-click clears   "
                     "Zoom: wheel   Pan: right-drag   Reset zoom: R")
        self.help_label = tk.Label(self.root, text=help_text, anchor="w", justify=tk.LEFT, fg="#555555",
                                   wraplength=1000)
        self.help_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        self.root.bind("<Configure>", self._on_root_configure, add="+")

        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 6))
        self.progress_bar_label = tk.Label(self.bottom_frame, text = "Measurement progress:")
        self.progress_bar_label.pack(side=tk.LEFT)
        self.progress = Progressbar(self.bottom_frame, orient=tk.HORIZONTAL, length=700)
        self.progress.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

    def _fit_window(self):
        """Size the window to its content, then let it grow."""
        # Size the window to fit the content, then let it grow (the canvas takes the extra space).
        self.root.update_idletasks()
        self.root.geometry(f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}")
        # Never shorter than the sidebar needs, or pack squeezes out its last widgets (buttons, logos).
        self.root.minsize(800, self.root.winfo_reqheight())
        self.root.resizable(True, True)


    def _on_root_configure(self, event):
        if event.widget is self.root:
            self.help_label.configure(wraplength=max(200, event.width - 20))

    # The old manual "sort by a substring of the filename" section
    # (start_sort/end_sort/imagename_in_sort_section) is gone: its widgets were
    # already commented out and its self.sort_txt_box never created, so all three
    # methods would have raised AttributeError if anything had called them. The
    # ordering they were meant to provide is now done automatically from each
    # frame's capture time in order_series().

    # --- Results and button state ------------------------------------------------

    def _has_results(self):
        return bool(self.frame_results_by_crop or self.cropped_filenames_by_crop)

    def _clear_results(self):
        """Forget every analysis result (frame results, crop filenames, masks, landmark readouts,
        germination, contour histories) and close the open preview windows, which hold those result
        lists by reference."""
        for window in list(self._analysis_windows):
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass
        self._analysis_windows = []
        self.frame_results_by_crop = {}
        self.cropped_filenames_by_crop = {}
        self.mask_store.clear()
        self.germination_detector = None
        self.reset_frame_accumulators()         # contour histories and landmark_store
        self._update_button_states()

    def _update_button_states(self):
        ready = bool(self.file_list) and bool(self.crop_boxes) and not self._busy
        normal, disabled = tk.NORMAL, tk.DISABLED
        self.btn_import_image["state"] = disabled if self._busy else normal
        for button in (self.btn_save_boxes, self.btn_delete_box, self.btn_clear_collar):
            button["state"] = normal if (self.crop_boxes and not self._busy) else disabled
        self.btn_load_boxes["state"] = normal if (self.file_list and not self._busy) else disabled
        self.button_start_analysis["state"] = normal if ready else disabled
        self.button_preview_seedlings["state"] = normal if ready else disabled
        self.button_kinematics["state"] = normal if ready else disabled
        self.button_export_results["state"] = normal if (ready and not self._exporting) else disabled

    def _update_status(self):
        parts = [f"{len(self.crop_boxes)} seedling(s)"]
        pairs = find_overlapping_pairs(self.crop_boxes)
        if pairs:
            parts.append("OVERLAP: " + ", ".join(f"{i + 1} & {j + 1}" for i, j in pairs))
        n_collars = sum(1 for collar in self.collar_overrides if collar is not None)
        parts.append(f"{n_collars} collar override(s)")
        if self.current_frame_name and self.file_list:
            idx = self.file_list.index(self.current_frame_name) + 1 if self.current_frame_name in self.file_list else 0
            parts.append(f"frame {idx}/{len(self.file_list)}: {self.current_frame_name}")
        self.status_var.set("   |   ".join(parts))

    # Crop-box editing / saving and the seed point live in ui/gui/crop_boxes.py, the scale in
    # ui/gui/scale.py, export in ui/gui/export.py, per-seedling windows in ui/gui/seedling_windows.py.

    # --- Analysis (background worker) ------------------------------------------

    def _threading_analysis(self):
        if self._has_results():
            if not messagebox.askyesno(
                    "Replace results?",
                    "Seedling results already exist (from a previous analysis or a preview). "
                    "Starting a new analysis replaces them -- including manual landmark edits and blanked frames.\n\n"
                    "Continue?", parent=self.root):
                return
        # Fresh run: a repeat would otherwise append onto the previous frame results.
        self._clear_results()
        # Disable the controls while the analysis runs (done here, on the main thread, since
        # start_analysis itself runs on the background thread from its first line).
        self._busy = True
        self._update_button_states()

        # Read image timestamps (and prompt for a fallback interval if needed) here,
        # on the main thread -- this may pop a modal dialog, which is only safe
        # before the background worker starts, not from inside it.
        self._collect_image_metadata()
        self._snapshot_tk_state()

        th = threading.Thread(target=self.start_analysis)
        th.start()

    def _snapshot_tk_state(self):
        """Read the Tk settings a worker needs here, on the main thread: workers must never
        call Tk (including Variable.get()) themselves."""
        self._debug_snapshot = bool(self.debug_var.get())

    def _debug_on(self):
        """Debug mode as snapshotted before the current worker started; falls back to reading
        the Tk variable when called on the main thread without a snapshot."""
        snap = getattr(self, "_debug_snapshot", None)
        if snap is not None:
            return snap
        return bool(self.debug_var.get())

    def _collect_image_metadata(self):
        # Capture times were already read (and used to order file_list) at import
        # time in add_files, so this reuses them instead of re-decoding every
        # file's headers a second time.
        self.metadata_list = [
            {"filename": image, "creation_time": self.capture_times.get(image)}
            for image in self.file_list
        ]

        untrustworthy = metadata_has_gaps(self.metadata_list)
        if untrustworthy:
            n_missing = sum(1 for m in self.metadata_list if m["creation_time"] is None)
            # None here means the user cancelled the dialog; the kinematics
            # window then falls back to plotting against the frame index.
            self.fallback_interval_minutes = simpledialog.askfloat(
                "No usable capture times",
                f"{n_missing} of {len(self.metadata_list)} images carry no capture time "
                "in their metadata\n(and file creation/modification times are not a "
                "substitute -- they record\nwhen this copy was written, not when the "
                "image was taken).\n\n"
                "Enter the interval between images (in minutes):",
                parent=self.root,
                minvalue=0.0,
            )
        else:
            self.fallback_interval_minutes = None

        # When the timestamps are untrustworthy they are discarded outright
        # rather than mixed with the interval -- see compute_time_deltas.
        self.time_deltas = compute_time_deltas(
            self.metadata_list,
            self.fallback_interval_minutes,
            trust_timestamps=not untrustworthy,
        )

        # Say which source the time axis came from; a wrong x-axis is otherwise
        # indistinguishable from a right one just by looking at the plot.
        if not untrustworthy:
            source = "embedded capture times"
        elif self.fallback_interval_minutes is not None:
            source = f"user-supplied interval of {self.fallback_interval_minutes} min/frame"
        else:
            source = "nothing (prompt cancelled) -- kinematics will use frame index"
        log.info(f"Time axis from {source}")

    def _pump_progress(self):
        """Runs on the main thread; drains ProgressReporter events from the worker thread
        and applies them to Tk widgets here instead of mutating them cross-thread."""
        try:
            for event in self.progress_reporter.drain():
                self._handle_progress_event(event)
        finally:
            # Re-armed even if an event handler raised, so one failure never stops the pump.
            self.root.after(100, self._pump_progress)

    def _handle_progress_event(self, event):
        if "value" in event:
            self.progress["value"] = event["value"]
        if "message" in event:
            self.progress_bar_label.configure(text=event["message"])
        error = event.get("error")
        if event.get("done"):
            self._busy = False
            self._update_button_states()
            self.progress_bar_label.configure(
                text="Analysis failed -- see the error message" if error else
                "Analysis complete -- use Preview seedling to review, or Export results when done")
        if event.get("export_done"):
            self._busy = False
            self._exporting = False
            self._update_button_states()
            if error:
                self.progress_bar_label.configure(text="Export failed -- see the error message")
            elif event.get("export_df") is not None:
                self._save_export(event["export_df"])
        if error:
            messagebox.showerror("DLhook", error, parent=self.root)

    def start_analysis(self):
        """Worker-thread entry point: always reports done (with the error, if any), so a failure
        never leaves the controls disabled."""
        error = None
        try:
            self._run_analysis()
        except Exception as exc:
            traceback.print_exc()
            error = f"Analysis failed: {exc}"
        finally:
            self.progress_reporter.report(done=True, error=error)

    def _run_analysis(self):
        # Initialize preprocessing
        preprocess_init = preprocess_images()

        # Progress bar setup
        n_images = len(self.file_list)

        # Process each image
        for image_n, image in enumerate(self.file_list):
            img_path = os.path.join(self.path, image)
            img_x = cv2.imread(img_path)

            # Update progress bar (0–30%)
            self.progress_reporter.report(value=(image_n / n_images) * 30)
            if img_x is None:
                # A corrupt or unreadable file: no crops. Its frames stay in the series as "no reading"
                # (_segment_paths records a None readout for a crop it could not read), so the frame
                # indices of every seedling still match file_list.
                log.warning(f"Cannot read {image}; its frame has no reading")
                continue

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

                if self.model_superres is not None:
                    # Opt-in super-resolution enhancement (4x), then downsized back
                    # to this crop's own native size -- sharpens detail
                    # without altering the crop's actual pixel dimensions.
                    # UNetInference tiles at native resolution (no fixed
                    # working size), so the saved crop must preserve its own
                    # box-derived size, not be forced to any fixed square.
                    orig_h, orig_w = crop.shape[:2]
                    crop_x4 = self.model_superres.enhance(crop)
                    crop = cv2.resize(crop_x4, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

                cv2.imwrite(os.path.join(self.path_save_x, f'{n}-crop-{_stem(image)}.png'), crop)
        
        self.run_apical_pipeline()


    def _segment_paths(self, paths):
        """Segment `paths` with the configured backend and store every
        label's masks into self.mask_store. Which models actually run (and
        why cotyledon_v3 / label "3" is skipped) lives in
        models/segmentation_backends.py -- this just delegates to whichever
        backend resolve_backend_name() picks. No chunking here: the caller
        decides whether/how to chunk its file list before calling this."""
        get_backend(resolve_backend_name()).predict_into(self.mask_store, paths)
        if resolve_angle_source() == "landmarks":
            # Same call sites as segmentation (batch, preview, export), so the readouts exist
            # exactly where the masks do. A failure leaves the store unchanged: process_single_frame
            # then falls back to the ellipse geometry for those frames.
            store = getattr(self, "landmark_store", None)
            if store is None:
                store = self.landmark_store = {}
            try:
                store.update(get_landmark_runner().predict_paths(paths))
            except Exception as exc:
                log.warning(f"Landmark readout failed ({exc}); using the ellipse angle source for these frames")
            else:
                for path in paths:      # an unreadable crop: no reading (a NaN frame, not a dropped one)
                    store.setdefault(os.path.basename(path), None)

    def run_apical_pipeline(self):
        image_crops_angles = {}
        image_crops_angles_max = {}

        # Construct filenames. This explicit list -- not os.listdir(data/images/)
        # -- is the work list handed to inference, so a leftover crop from an
        # earlier "Preview seedling" run can't be silently re-segmented here.
        self.cropped_sorted_filenames = [
            f"{crop_n}-crop-{_stem(filename)}.png"
            for crop_n in range(len(self.transformed_mid_points))
            for filename in self.file_list
        ]

        total_files = len(self.cropped_sorted_filenames)

        self.reset_frame_accumulators()

        # Inference and per-frame processing are INTERLEAVED in chunks rather
        # than run as three whole-dataset passes followed by a frame loop. All
        # three models stay resident (the backend's own process-wide cache,
        # see models/segmentation_backends.py), so alternating between them
        # per chunk costs no reloads -- and it means only one chunk's worth
        # of raw masks is ever held at once, instead of every mask for the
        # whole run.
        for start in range(0, total_files, IMAGE_CHUNK):
            chunk = self.cropped_sorted_filenames[start:start + IMAGE_CHUNK]
            chunk_paths = [crop_path(f) for f in chunk]

            self._segment_paths(chunk_paths)

            for offset, file_name in enumerate(chunk):
                idx = start + offset
                self.progress_reporter.report(value=30 + int((idx / total_files) * 70))
                log.debug("Processing %s", file_name)

                # Filenames are "{crop_id}-crop-{original_name}.png" (see
                # cropped_sorted_filenames above) -- split on the first "-"
                # rather than indexing the first character, which silently
                # broke for crop_id >= 10.
                crop_id = int(file_name.split("-", 1)[0])
                result = self.process_single_frame(file_name, crop_id)
                if result is None:
                    continue

                hook = result["hook"]
                seed_ids = result["seed_ids"]
                angle_dict = result["angle_dict"]

                if hook is not None:    # None for the landmark angle source (no ellipse geometry)
                    hook.save(data_path("final_prediction", file_name, make=True))
                log.debug(f"Angles found for {seed_ids}: {angle_dict}")

                # Mirrors segment_single_seedling/SeedlingAnalysisWindow's own
                # bookkeeping, so a seedling already segmented by this batch pass
                # isn't redundantly re-segmented by "Preview seedling"/"Export results".
                self.frame_results_by_crop.setdefault(crop_id, []).append(result)
                self.cropped_filenames_by_crop.setdefault(crop_id, []).append(file_name)

                # Save max angles (this part stays the same)
                image_crops_angles[crop_id] = angle_dict
                max_dict = image_crops_angles_max.setdefault(crop_id, {})
                for sid, angle in angle_dict.items():
                    if isinstance(angle, (int, float)) and not np.isnan(angle):
                        if sid not in max_dict or angle > max_dict.get(sid, float('-inf')):
                            max_dict[sid] = angle

            # Germination masks (label "4") are consumed as contours by
            # GerminationDetector, so the raw prediction can be freed as soon
            # as this chunk's frames are processed. (The analysis window's germ overlay reads
            # result["germ_mask"], its own thresholded copy, so it still renders
            # after this.)
            # Cotyledon ("1") and hypocotyl ("2") raw masks are deliberately
            # KEPT for the whole session (trait masks are re-read from the store).
            # (kept when dumping, so the dump still contains every label)
            if not self.dump_masks:
                # (the "hypocotyl_binary" trait mask is kept like "1"/"2": the analysis window and
                # landmark edits re-read it for the hypocotyl traits)
                self.mask_store.discard_raw(chunk, labels=("4",))

        # Germination time-zero per seedling, from the germ_v1 mask time series
        # process_single_frame accumulated above, keyed by crop_id.
        for crop_id in range(len(self.transformed_mid_points)):
            self._ensure_germination_detected(crop_id)
        log.info(f"Germination time-zero per seedling: {self.germination_detector.germination_frame}")

        # Angle-through-time reconstruction (utils/angle_timeseries.py), one
        # seedling at a time -- must run after every frame's raw per-frame
        # angle has been computed above, same requirement as germination
        # detection just above it.
        for crop_id in range(len(self.transformed_mid_points)):
            self._reconstruct_series_for_crop(crop_id)

        # Opt-in (DLHOOK_DUMP_MASKS=1): write the session's masks out in the
        # legacy 0 = foreground on-disk convention, so they stay directly
        # comparable with previously exported ground-truth masks.
        if self.dump_masks:
            self.mask_store.dump(data_path("predict", make=True))

    def reset_frame_accumulators(self, crop_id=None):
        """Clears the per-crop_id contour history process_single_frame builds
        up as it walks a seedling's frames in order (ApicalHook/GerminationDetector
        need it in time order, so a fresh run mustn't append onto a stale one)."""
        if crop_id is None:
            self.cotyl_time_series_by_crop = {}
            self.germ_time_series_by_crop = {}
            self.landmark_store = {}
        else:
            self.cotyl_time_series_by_crop[crop_id] = []
            self.germ_time_series_by_crop[crop_id] = []
            # Landmark readouts of this crop's frames ("{crop_id}-crop-..."): parse, never index [0].
            store = getattr(self, "landmark_store", None) or {}
            for name in [n for n in store if n.split("-", 1)[0] == str(crop_id)]:
                del store[name]

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
        # Learned detector first (models/segmentation_backends.py: resolve_germination_method);
        # the area-near-the-seed rule below is the fallback and DLHOOK_GERMINATION=rule.
        if self._detect_germination_learned(crop_id):
            return
        frame_results = getattr(self, "frame_results_by_crop", {}).get(crop_id, [])
        if frame_results and not any(r is not None and r.get("germ_mask") is not None for r in frame_results):
            # No radicle ("4") mask exists for this seedling (the default multiclass backend does
            # not segment it), so the area rule would run on empty contours and report a bogus
            # "no germination". Record it as unknown instead (get_time_zero -> None, empty CSV cell).
            self.germination_detector.germination_frame[crop_id] = None
            if not getattr(self, "_warned_germination_unknown", False):
                self._warned_germination_unknown = True
                log.warning("Germination time-zero is unknown: the learned detector is unavailable "
                      "and no radicle mask was segmented (area rule needs DLHOOK_SEG_BACKEND=binary).")
            return
        frame_contours = self.germ_time_series_by_crop.get(crop_id, [])
        box = self.crop_boxes[crop_id]
        # The seed point of the last frame (collar override, else that frame's landmark collar);
        # without either, the crop's bottom-centre -- where a radicle emerging from the seed coat
        # is expected relative to the crop.
        files = self.cropped_filenames_by_crop.get(crop_id) or []
        last_file = files[-1] if files else (
            f"{crop_id}-crop-{_stem(self.file_list[-1])}.png" if getattr(self, "file_list", None) else None)
        seed_point = self._seed_point(crop_id, last_file)
        if seed_point is None:
            seed_point = (box["half_w"], 2 * box["half_h"] - 1)
        # This seedling's own crop's pixel size -- crops keep their native,
        # box-derived dimensions (not resized to a shared fixed working
        # size), so the proximity radius/area threshold must scale per
        # seedling instead of assuming one global crop size.
        crop_size = (2 * box["half_w"] + 2 * box["half_h"]) / 2
        self.germination_detector.detect(crop_id, frame_contours, seed_point, crop_size)
        # The thresholds are uncalibrated (no ground-truth germination timing
        # exists to fit them against), so with Debug on, log what detect()
        # actually measured -- all-zero areas mean the germ mask itself is
        # empty, nonzero-but-below-threshold means the threshold needs tuning.
        if self._debug_on():
            log.debug(f"germination {self.germination_detector.describe(crop_id)}")

    def _detect_germination_learned(self, crop_id):
        """Time-zero from the learned detector over this seedling's crops, in the same frame order
        as its frame results. Returns False (so the caller falls back to the rule) when the method
        is 'rule', there are no crops yet, or the learned run fails."""
        if resolve_germination_method() != "learned":
            return False
        files = self.cropped_filenames_by_crop.get(crop_id, [])
        if not files:
            return False
        try:
            onset, probs = get_germination_runner().onset([crop_path(f) for f in files])
        except Exception as exc:                      # never let time-zero detection sink an analysis
            log.warning(f"learned germination failed for seedling {crop_id + 1} ({exc}); using the area rule")
            return False
        self.germination_detector.record_learned(crop_id, onset, probs)
        if self._debug_on():
            log.debug(f"germination {self.germination_detector.describe(crop_id)}")
        return True

    def _reconstruct_series_for_crop(self, crop_id):
        """Runs the angle-through-time temporal reconstruction (see
        utils/angle_timeseries.py) for one seedling, over whatever frames
        process_single_frame has produced so far in
        self.frame_results_by_crop[crop_id]. Same idempotent,
        callable-from-anywhere contract as _ensure_germination_detected
        above -- safe to call on a partial preview now and a fuller
        export/batch run later, since each call re-reconstructs from the
        original per-frame geometric angle rather than an already-
        reconstructed one (see the raw_angle_dict capture below).

        Rewrites each frame's angle_dict/state_dict in place, so callers
        holding a reference to the same result dicts (e.g. an open
        SeedlingAnalysisWindow) see the update immediately. A frame whose
        state is already "Manual" is a fixed anchor -- reconstruct_series
        already leaves it untouched, so no special-casing is needed here
        beyond feeding it in as such.

        Convention: AngleCalculator.compute_biological_angle emits ~0 when
        closed, growing as the hook opens, but the whole reconstruction +
        the export's bio_angle column + the user's thresholds use the
        opposite "bio" convention (180 = closed; see utils/angle_timeseries.py).
        So each automated per-frame reading is converted to bio via
        `180 - raw` before reconstruction. raw_angle_dict keeps the original
        (pre-conversion, 0 = closed) geometric reading, which is what the
        export's raw_angle column shows. Manual overrides are already stored
        in bio (see SeedlingAnalysisWindow._apply_landmarks), so
        they're fed through unchanged.
        """
        frame_results = self.frame_results_by_crop.get(crop_id)
        if not frame_results:
            return
        seed_id = crop_id + 1

        # Series algorithm: any landmark-source frame => the landmark reconstruction (the ellipse
        # branch DP would flip readings that were already right). Frames are still read by their own
        # source, so a mixed list (a few ellipse fallbacks) is fed through the same path.
        if any(r is not None and r.get("angle_source") == "landmarks" for r in frame_results):
            self._reconstruct_landmark_series_for_crop(frame_results, seed_id)
            return

        def _to_bio(v):
            return 180.0 - v if isinstance(v, (int, float)) and not np.isnan(v) else None

        bio_inputs = []
        is_manual = []
        for result in frame_results:
            if result is None:
                bio_inputs.append(None)
                is_manual.append(False)
                continue
            manual = result.get("state_dict", {}).get(seed_id) == "Manual"
            is_manual.append(manual)
            if manual:
                # Manual overrides are already in bio convention -- feed as-is.
                bio_inputs.append(result["angle_dict"].get(seed_id))
                continue
            # Captured once per frame, on first reconstruction -- so a
            # repeat call (partial preview, then a fuller export) always
            # reconstructs from the same untouched geometric reading
            # instead of re-smoothing an already-reconstructed value. A
            # re-segmentation returns fresh result dicts with no
            # raw_angle_dict yet, so their readings become the new baseline.
            raw_dict = result.setdefault("raw_angle_dict", {})
            if seed_id not in raw_dict:
                raw_dict[seed_id] = result["angle_dict"].get(seed_id)
            bio_inputs.append(_to_bio(raw_dict[seed_id]))

        angles_out, states_out = reconstruct_series(bio_inputs, is_manual=is_manual)

        for result, angle, state, manual in zip(frame_results, angles_out, states_out, is_manual):
            if result is None or manual:
                continue
            result["angle_dict"][seed_id] = angle
            result.setdefault("state_dict", {})[seed_id] = state
            sync_angle_and_state_lists(result)

    def _reconstruct_landmark_series_for_crop(self, frame_results, seed_id):
        """Landmark-source branch of _reconstruct_series_for_crop (same contract: idempotent, raw
        reading captured once into raw_angle_dict, "Manual" frames are fixed anchors).

        The raw reading is theta (0 = closed, the same convention as the ellipse path), so the bio
        value is 180 - theta -- produced inside reconstruct_landmark_series, the single conversion.
        Input comes from raw_angle_dict (captured from result["landmarks"] on first sight), never
        from the already-reconstructed angle_dict. Manual frames keep their stored bio value; those
        at or below 180 also feed the series as an anchor reading (theta = 180 - bio) so neighbouring
        gaps interpolate against them. The robust reconstruction (utils/landmark_timeseries) rejects
        readings far from their neighbours and smooths the rest; Manual frames are never changed and
        anchor it. Each landmark readout gets "series_edit" ("" / "rejected" / "interpolated"), re-derived
        on every call. States come from classify_states."""
        readouts = []
        is_manual = []
        manual_bio = []
        gated = []      # readability-gated frames: no information, never interpolated, state "Unreadable"
        for result in frame_results:
            gated.append(False)
            if result is None:
                readouts.append(None)
                is_manual.append(False)
                manual_bio.append(None)
                continue
            manual = result.get("state_dict", {}).get(seed_id) == "Manual"
            is_manual.append(manual)
            if manual:
                bio = result["angle_dict"].get(seed_id)
                manual_bio.append(bio)
                ok = isinstance(bio, (int, float)) and not np.isnan(bio) and bio <= 180.0
                readouts.append({"theta": 180.0 - bio, "peak": 1.0, "manual": True} if ok else None)
                continue
            manual_bio.append(None)
            raw_dict = result.setdefault("raw_angle_dict", {})
            readout = (result.get("landmarks") or {}).get(seed_id)
            from_landmarks = result.get("angle_source") == "landmarks"
            if from_landmarks and readout and readout.get("gated"):
                # raw_angle_dict keeps the reading actually used (NaN); the model's theta stays
                # in readout["gated_theta"] for reference.
                raw_dict[seed_id] = float("nan")
                gated[-1] = True
                readouts.append({"theta": None, "gated": True})
                continue
            if seed_id not in raw_dict:
                if from_landmarks:
                    theta = readout.get("theta") if readout else None
                    raw_dict[seed_id] = float(theta) if theta is not None else float("nan")
                else:       # an ellipse fallback frame inside a landmark series
                    raw_dict[seed_id] = result["angle_dict"].get(seed_id)
            raw = raw_dict[seed_id]
            if isinstance(raw, (int, float)) and not np.isnan(raw):
                peak = readout.get("peak", 1.0) if (readout and from_landmarks) else 1.0
                mp = readout.get("measurable_p") if (readout and from_landmarks) else None
                readouts.append({"theta": float(raw), "peak": peak, "measurable_p": mp})
            else:
                readouts.append(None)

        bio, edits = reconstruct_landmark_series_detail(readouts)
        bio = [manual_bio[i] if is_manual[i] else v for i, v in enumerate(bio)]
        for result, edit, manual in zip(frame_results, edits, is_manual):
            readout = (result.get("landmarks") or {}).get(seed_id) if result else None
            if readout:
                readout["series_edit"] = "" if manual else edit
        states = classify_states(bio, is_manual)
        self._flag_landmark_series(frame_results, seed_id, is_manual, gated)

        for result, angle, state, manual, gate in zip(frame_results, bio, states, is_manual, gated):
            if result is None or manual:
                continue
            result["angle_dict"][seed_id] = np.nan if (angle is None or np.isnan(angle)) else float(angle)
            result.setdefault("state_dict", {})[seed_id] = UNREADABLE_STATE if gate else state
            sync_angle_and_state_lists(result)

    @staticmethod
    def _flag_landmark_series(frame_results, seed_id, is_manual, gated):
        """Temporal consistency flags (utils/landmark_temporal.py) written onto each frame's landmark
        readout: "temporal_flag", "temporal_collar_dev", "collar_ref". Flag only -- angles and states
        are untouched. Derived afresh on every reconstruction, so a manual edit clears stale flags."""
        points, readouts = [], []
        for result, manual, gate in zip(frame_results, is_manual, gated):
            readout = (result.get("landmarks") or {}).get(seed_id) if result else None
            readouts.append(readout)
            usable = readout and not gate and (manual or result.get("angle_source") == "landmarks")
            points.append({"junction": readout.get("junction"), "collar": readout.get("collar")} if usable else None)
        for readout, info in zip(readouts, temporal_flags(points, is_manual)):
            if readout:
                readout["temporal_flag"] = info["flag"]
                readout["temporal_collar_dev"] = info["collar_dev"]
                readout["collar_ref"] = info["collar_ref"]

    def process_single_frame(self, file_name, crop_id, frame_index=None):
        """
        Mask load -> threshold -> erode/dilate -> contour -> ApicalHook.process()
        for one already-cropped, already-segmented frame. Shared by the full
        batch pipeline (run_apical_pipeline) and the Phase 5 per-seedling
        preview window, so both compute angles identically.

        Masks are loaded from self.mask_store.

        By default (frame_index=None) this appends onto
        self.cotyl_time_series_by_crop[crop_id]/germ_time_series_by_crop[crop_id],
        for callers walking a seedling's frames in order from the start
        (reset_frame_accumulators(crop_id) first for a fresh run). Passing
        frame_index instead REPLACES that one entry in place, reprocessing a single
        already-processed frame without disturbing the frames around it or
        duplicating history entries.

        Returns None if the image or a required mask is missing, else a dict
        with the ApicalHook instance, angle results, and the frame's masks
        (for a caller that wants to render an overlay preview).
        """
        image_path = crop_path(file_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Masks come from the in-memory MaskStore. They are already in the
        # 255 = foreground convention, so none of the bitwise_not calls the
        # old on-disk format required are needed here.
        img_cotyl = self.mask_store.get(file_name, "1")
        img_hypo = self.mask_store.get(file_name, "2")
        img_germ = self.mask_store.get(file_name, "4")

        # Angle source for THIS frame: the landmark readout when the source is "landmarks" and the
        # readout exists for the frame (None = the model found no hook: NaN angle, still landmarks);
        # otherwise the legacy ellipse path below, unchanged.
        landmark_store = getattr(self, "landmark_store", None) or {}
        if resolve_angle_source() == "landmarks" and file_name in landmark_store:
            return self._process_landmark_frame(
                file_name, crop_id, frame_index, landmark_store[file_name],
                img_cotyl, img_hypo, img_germ)

        if image is None or img_cotyl is None or img_hypo is None:
            log.warning(f"Skipping {file_name}: image or masks not found.")
            return None
        return self._process_ellipse_frame(file_name, crop_id, frame_index, image, img_cotyl, img_hypo, img_germ)

    def _process_ellipse_frame(self, file_name, crop_id, frame_index, image, img_cotyl, img_hypo, img_germ):
        """The legacy ellipse angle source for one frame (see process_single_frame)."""
        seed_ids = [crop_id + 1]
        seed_point = self._seed_point(crop_id, file_name)
        cotyl_mask, hypo_mask, germ_mask = self._clean_ellipse_masks(img_cotyl, img_hypo, img_germ, seed_point)
        # ApicalHook (one seedling per crop, so the point only labels the match and is drawn on the
        # overlay) gets the crop's bottom-centre when there is no collar.
        seed_points = [seed_point if seed_point is not None
                       else self._bottom_centre(img_hypo.shape[1], img_hypo.shape[0])]

        # Extract contours. The >= 5 point floor is a cv2.fitEllipse
        # prerequisite in the cotyledon/hypocotyl angle path; germination only
        # ever takes contourArea, and a small emerging radicle can legitimately
        # trace fewer than 5 points, so germ is not filtered that way.
        cotyl_contours = [c for c in cv2.findContours(cotyl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
        hypo_contours = [c for c in cv2.findContours(hypo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if len(c) >= 5]
        germ_contours = (list(cv2.findContours(germ_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
                          if germ_mask is not None else [])
        cotyl_history_for_hook, germ_history_for_hook = self._record_contours(
            crop_id, frame_index, cotyl_contours, germ_contours)

        log.debug(f"Seed points: {seed_points}")
        log.debug(f"Cotyl contours: {len(cotyl_contours)} | Hypo contours: {len(hypo_contours)}")

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
            "angle_source": "ellipse",
            "traits": {},        # junction unknown on this path, so no hypocotyl traits
        }

    def _clean_ellipse_masks(self, img_cotyl, img_hypo, img_germ, seed_point):
        """(cotyl_mask, hypo_mask, germ_mask) for the ellipse path: thresholded, cut at the seed line,
        opened and zoomed out (germ only thresholded; None when there is no "4" mask)."""
        # Process masks
        _, bin_cotyl = cv2.threshold(img_cotyl, 127, 255, cv2.THRESH_BINARY)
        _, bin_hypo = cv2.threshold(img_hypo, 127, 255, cv2.THRESH_BINARY)
        # No bitwise_not on germ: the MaskStore hands out 255 = foreground for
        # every label. Re-inverting germ here (which the old on-disk pipeline
        # effectively did, and which this code reproduced deliberately) made
        # every germ contour describe the crop's BACKGROUND -- a single
        # near-full-frame rectangle, whose 4-point contour the >=5-point filter
        # below then dropped. That is why germination detection returned None
        # for every seedling.
        bin_germ = cv2.threshold(img_germ, 127, 255, cv2.THRESH_BINARY)[1] if img_germ is not None else None

        # The seed line masks everything below the collar; with no collar (no override, no
        # landmark readout) nothing is masked.
        above_mask = self._above_seed_mask(img_hypo.shape, seed_point)

        cotyl_mask = cv2.bitwise_and(bin_cotyl, bin_cotyl, mask=above_mask)
        hypo_mask = cv2.bitwise_and(bin_hypo, bin_hypo, mask=above_mask)

        # Define kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Apply erosion and dilation
        cotyl_mask = cv2.erode(cotyl_mask, kernel, iterations=1)
        cotyl_mask = cv2.dilate(cotyl_mask, kernel, iterations=1)

        hypo_mask = cv2.erode(hypo_mask, kernel, iterations=1)
        hypo_mask = cv2.dilate(hypo_mask, kernel, iterations=1)

        cotyl_mask = zoom_out_mask(cotyl_mask, scale=0.98)
        hypo_mask = zoom_out_mask(hypo_mask, scale=0.98)
        # germ is NOT zoomed out: that helper shrinks the mask about the crop's
        # center to detach blobs from the border, which displaces the germ blob
        # relative to the seed coat point -- and the distance between those two
        # is exactly what GerminationDetector measures.
        germ_mask = bin_germ
        return cotyl_mask, hypo_mask, germ_mask

    def _record_contours(self, crop_id, frame_index, cotyl_contours, germ_contours):
        """Append (frame_index None) or replace this frame's contours in the seedling's history; returns
        the (cotyledon, germ) histories ApicalHook should see, ending at this frame."""
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
        return cotyl_history_for_hook, germ_history_for_hook

    def _trait_mask(self, file_name, fallback=None):
        """The mask the hypocotyl traits are measured on: the binary hypocot_v5 mask when the backend
        ran it ("hypocotyl_binary", off by default since the v2 four-class model), else label "2" (the
        multiclass hypocotyl, which reaches the junction), else `fallback`. Shared by the pipeline and the
        analysis window's landmark edits."""
        mask = self.mask_store.get(file_name, HYPOCOTYL_BINARY_LABEL)
        if mask is None:
            mask = self.mask_store.get(file_name, "2")
        return fallback if mask is None else mask

    def _process_landmark_frame(self, file_name, crop_id, frame_index, readout,
                                img_cotyl, img_hypo, img_germ):
        """process_single_frame for the landmark angle source. The angle needs no mask, so a frame
        is NOT dropped when masks are missing: cotyl/hypo/germ masks and "hook" are simply None
        (consumers guard for that). angle_dict holds the raw theta (0 = closed; the single 180 - raw
        conversion lives in _reconstruct_series_for_crop), NaN when there is no readout."""
        seed_id = crop_id + 1
        seed_point = self._seed_point(crop_id, file_name)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        def _clean(raw, above):
            if raw is None:
                return None
            m = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)[1]
            if above is not None:
                m = cv2.bitwise_and(m, m, mask=above)
            m = cv2.dilate(cv2.erode(m, kernel, iterations=1), kernel, iterations=1)
            return zoom_out_mask(m, scale=0.98)

        ref = next((m for m in (img_hypo, img_cotyl, img_germ) if m is not None), None)
        above = self._above_seed_mask(ref.shape, seed_point) if ref is not None else None
        cotyl_mask = _clean(img_cotyl, above)
        hypo_mask = _clean(img_hypo, above)
        germ_mask = cv2.threshold(img_germ, 127, 255, cv2.THRESH_BINARY)[1] if img_germ is not None else None

        def _contours(mask, min_points):
            if mask is None:
                return []
            return [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(c) >= min_points]

        cotyl_contours = _contours(cotyl_mask, 5)
        germ_contours = _contours(germ_mask, 0)
        cotyl_history = self.cotyl_time_series_by_crop.setdefault(crop_id, [])
        germ_history = self.germ_time_series_by_crop.setdefault(crop_id, [])
        if frame_index is None:
            cotyl_history.append(cotyl_contours)
            germ_history.append(germ_contours)
        else:
            cotyl_history[frame_index] = cotyl_contours
            germ_history[frame_index] = germ_contours

        theta = readout.get("theta") if readout else None
        # Readability gate (utils/landmark_gate): an unreadable frame has no angle (NaN) but keeps
        # its readout, with the model's theta in "gated_theta", so the window can show it.
        gated = bool(readout and readout.get("unreadable") and gate_enabled())
        angle = float(theta) if theta is not None and not gated else float("nan")
        landmarks = {seed_id: dict(readout, edit="model") if readout else None}
        if gated:
            landmarks[seed_id].update(gated=True, gated_theta=float(theta) if theta is not None else None)

        trait_mask = self._trait_mask(file_name, img_hypo)
        junction = readout.get("junction") if readout else None
        collar = readout.get("collar") if readout else None
        traits = {seed_id: hypocotyl_traits(trait_mask, junction, collar)}

        return {
            "hook": None,
            "angle_dict": {seed_id: angle},
            "angle_list": ['-' if np.isnan(angle) else round(angle)],
            "state_dict": {seed_id: "-"},     # placeholder; classify_states fills it in the reconstruction
            "state_list": ["-"],
            "seed_ids": [seed_id],
            "cotyl_mask": cotyl_mask,
            "hypo_mask": hypo_mask,
            "germ_mask": germ_mask,
            "angle_source": "landmarks",
            "landmarks": landmarks,
            "traits": traits,
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

            out_name = f"{crop_id}-crop-{_stem(image)}.png"
            cv2.imwrite(os.path.join(self.path_save_x, out_name), crop)
            cropped_filenames.append(out_name)

        return cropped_filenames

    def segment_single_seedling(self, crop_id, progress_reporter=None):
        """
        Crops (if needed) and runs UNetInference scoped to just this one
        seedling's own cropped frames, filling the shared mask_store same as
        the batch pipeline -- lets a Phase 5 preview window spot-check one
        seedling without running inference over every other seedling too.
        Runs on a background thread; progress_reporter must be a
        ProgressReporter the caller polls on the main thread.
        """
        reporter = progress_reporter or self.progress_reporter

        reporter.report(message=f"Cropping seedling {crop_id + 1}...")
        self.reset_frame_accumulators(crop_id)
        cropped_filenames = self.crop_single_seedling(crop_id)
        file_paths = [crop_path(f) for f in cropped_filenames]

        reporter.report(message="Running segmentation models...")
        self._segment_paths(file_paths)

        reporter.report(message="Computing angles...")
        return cropped_filenames

    def add_files(self):
        """Pick an image folder and load its frames in acquisition order.

        A new folder is a new series: the boxes drawn on the previous one (and any results) are
        dropped, after a confirmation when there is something to lose."""
        path = askdirectory()
        if not path:
            return
        file_list, capture_times = order_series(path, _list_image_files(path))
        if not file_list:
            messagebox.showwarning(
                "No images", f"No image files ({', '.join(VALID_IMAGE_EXTENSIONS)}) found in:\n{path}",
                parent=self.root)
            return
        if self.crop_boxes or self._has_results():
            if not messagebox.askyesno(
                    "Open a new folder?",
                    "Opening another folder clears the current boxes and any results.\n\n"
                    "(Use Save boxes first to keep them.)  Continue?", parent=self.root):
                return

        self._clear_results()
        self.path = path
        self.series_name = os.path.basename(os.path.normpath(path))
        # Order the series BEFORE the listbox is filled, so the listbox and file_list agree.
        self.file_list, self.capture_times = file_list, capture_times
        self.filenames_listbox = True
        self.current_frame_name = ""
        self.width1 = self.height1 = 0           # the first frame shown sets the image size
        self._commit_boxes([], [])

        self.listbox.delete(0, tk.END)
        for file_n in self.file_list:
            self.listbox.insert(tk.END, file_n)

        # Select and show the last image (the seedlings are largest there)
        last_index = len(self.file_list) - 1
        self.listbox.selection_clear(0, tk.END)
        self.listbox.select_set(last_index)
        self.listbox.see(last_index)
        self.show_frame(self.file_list[last_index], reset_view=True)
        self._push_boxes_to_canvas()
        self._update_button_states()
        self._update_status()

    def _on_frame_selected(self, event=None):
        selection = self.listbox.curselection()
        if not selection or not self.path:
            return
        name = self.listbox.get(selection[0])
        if name != self.current_frame_name:
            self.show_frame(name)

    def show_frame(self, name, reset_view=False):
        """Show frame `name` of the open folder on the canvas at native resolution. The boxes and
        the zoom stay as they are (the zoom only resets when the image size changes)."""
        image = cv2.imread(os.path.join(self.path, name), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Cannot open image", f"Failed to read image file:\n{name}", parent=self.root)
            return False
        height, width = image.shape[:2]
        size_changed = (width, height) != (self.width1, self.height1)
        self.height1, self.width1 = height, width
        self.canvas.set_image(image, reset_view=reset_view or size_changed)
        self.current_frame_name = name
        self._update_status()
        return True



#Remove image files at exit of software
def remove_files_exit():
    RemoveData()

atexit.register(remove_files_exit)

if __name__== '__main__':
    remove_files_exit()
    root=tk.Tk()
    gui=Gui(root)
    root.mainloop()