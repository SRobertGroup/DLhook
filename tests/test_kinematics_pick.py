"""Clicking a point in the kinematics window opens that seedling's analysis window at that frame."""
import types

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


def _result(seed_id, angle, state="Closed"):
    return {"angle_dict": {seed_id: angle}, "state_dict": {seed_id: state}}


def _fake_gui(root):
    opened = []
    gui = types.SimpleNamespace(
        root=root,
        file_list=["f0.png", "f1.png", "f2.png", "f3.png"],
        time_deltas=[],
        germination_detector=None,
        # seedling 1: frame 1 has no angle, so plotted point k is not frame k
        frame_results_by_crop={0: [_result(1, 170.0), _result(1, np.nan), _result(1, 120.0), _result(1, 90.0)],
                               1: [_result(2, 175.0), None, _result(2, 150.0), _result(2, 100.0)]},
        open_seedling_frame=lambda crop_id, frame_index=None: opened.append((crop_id, frame_index)),
    )
    return gui, opened


def _pick(window, crop_id, plotted_index):
    artist = next(a for a, (cid, _) in window._pick_targets.items() if cid == crop_id)
    event = types.SimpleNamespace(artist=artist, ind=np.array([plotted_index]), mouseevent=None)
    window._on_pick(event)


def test_pick_maps_the_plotted_point_back_to_its_frame(root):
    from ui.kinematics_window import KinematicsWindow

    gui, opened = _fake_gui(root)
    window = KinematicsWindow(gui)
    try:
        _pick(window, 0, 1)          # second plotted point of seedling 1 = frame 2 (frame 1 was NaN)
        _pick(window, 1, 2)          # third plotted point of seedling 2 = frame 3 (frame 1 had no result)
        assert opened == [(0, 2), (1, 3)]
        window.refresh()             # a refresh rebuilds the targets, still one per plotted seedling
        assert sorted(cid for cid, _ in window._pick_targets.values()) == [0, 1]
    finally:
        window.destroy()


def test_open_seedling_frame_reuses_an_open_window_and_jumps_to_the_frame(root, monkeypatch):
    import seedling_measurement as sm
    from ui.gui import seedling_windows

    made = []

    class FakeWindow:
        def __init__(self, gui, crop_id):
            self.crop_id, self.frames, self.alive = crop_id, [], True
            made.append(self)

        def winfo_exists(self):
            return self.alive

        def show_frame(self, i):
            self.frames.append(i)

        def deiconify(self):
            pass

        lift = focus_force = deiconify

    monkeypatch.setattr(seedling_windows, "SeedlingAnalysisWindow", FakeWindow)
    gui = sm.Gui.__new__(sm.Gui)
    gui._analysis_windows = []

    w1 = gui.open_seedling_frame(0, 4)
    w2 = gui.open_seedling_frame(0, 7)
    w3 = gui.open_seedling_frame(1)
    assert w1 is w2 and w1.frames == [4, 7] and w3 is not w1 and w3.frames == []
    w1.alive = False                 # closed by the user: a new one is made
    w4 = gui.open_seedling_frame(0, 1)
    assert w4 is not w1 and w4.frames == [1] and len(made) == 3


def test_analysis_window_edits_redraw_the_open_kinematics_plot(root):
    """An edit notified through Gui._notify_results_changed redraws the open plot (once per burst);
    reopening raises the same window; a closed window is forgotten."""
    from seedling_measurement import Gui
    from ui.kinematics_window import KinematicsWindow

    gui, _ = _fake_gui(root)
    for name in ("_open_kinematics_window", "_live_kinematics_window", "_notify_results_changed"):
        setattr(gui, name, getattr(Gui, name).__get__(gui))
    gui._notify_results_changed()                       # nothing open: no-op
    gui._open_kinematics_window()
    window = gui.kinematics_window
    try:
        assert isinstance(window, KinematicsWindow)
        drawn = []
        real = window.refresh
        window.refresh = lambda: (drawn.append(1), real())
        gui.frame_results_by_crop[0][0]["angle_dict"][1] = 60.0          # an edit in the analysis window
        gui._notify_results_changed()
        first = window._refresh_job
        gui._notify_results_changed()                                   # burst: the pending redraw is replaced
        assert window._refresh_job is not None and window._refresh_job != first
        window.after_cancel(window._refresh_job)
        window._run_scheduled_refresh()                                 # what the timer runs
        assert drawn == [1] and window._refresh_job is None
        offsets = next(a for a, (cid, _) in window._pick_targets.items() if cid == 0).get_offsets()
        assert 60.0 in offsets[:, 1]                                    # the edited angle is plotted
        gui._open_kinematics_window()                                   # reopen: same window
        assert gui.kinematics_window is window
    finally:
        window.destroy()
    assert gui._live_kinematics_window() is None                       # closed: forgotten


def test_analysis_window_handlers_notify_the_gui():
    """Apply landmarks / Accept reading (_commit_manual), Previous collar, Blank, germination buttons and
    re-segmentation all end in _results_changed."""
    import inspect

    from ui.analysis_window import SeedlingAnalysisWindow

    for name in ("_commit_manual", "_use_previous_collar", "_blank_angle", "_set_germination_frame", "_apply_germination_to_all",
                 "_clear_germination_frame", "_handle_progress_event"):
        assert "self._results_changed()" in inspect.getsource(getattr(SeedlingAnalysisWindow, name)), name
    calls = []
    fake = types.SimpleNamespace(gui=types.SimpleNamespace(_notify_results_changed=lambda: calls.append(1)))
    SeedlingAnalysisWindow._results_changed(fake)
    SeedlingAnalysisWindow._results_changed(types.SimpleNamespace(gui=types.SimpleNamespace()))   # older fakes
    assert calls == [1]
