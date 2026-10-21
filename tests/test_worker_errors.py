"""A failing worker thread always reports done (with its error), and the progress pumps survive
a failing event handler -- so the controls are never left disabled (audit M-1)."""
from types import SimpleNamespace

import seedling_measurement as sm
from ui.gui import export as gui_export
from utils.gui_thread_safety import ProgressReporter


def _gui(**attrs):
    g = sm.Gui.__new__(sm.Gui)
    g.progress_reporter = ProgressReporter()
    for k, v in attrs.items():
        setattr(g, k, v)
    return g


def _boom(*_a, **_k):
    raise RuntimeError("model file is corrupt")


def test_analysis_worker_reports_done_with_the_error(monkeypatch):
    g = _gui()
    monkeypatch.setattr(g, "_run_analysis", _boom)
    g.start_analysis()                                   # must not raise
    events = g.progress_reporter.drain()
    assert events[-1]["done"] is True and "model file is corrupt" in events[-1]["error"]


def test_analysis_worker_reports_done_without_error_on_success(monkeypatch):
    g = _gui()
    monkeypatch.setattr(g, "_run_analysis", lambda: None)
    g.start_analysis()
    assert g.progress_reporter.drain()[-1] == {"done": True, "error": None}


def test_export_worker_reports_export_done_with_the_error(monkeypatch):
    g = _gui()
    monkeypatch.setattr(g, "_build_export_df", _boom)
    g._run_export_results()
    last = g.progress_reporter.drain()[-1]
    assert last["export_done"] is True and last["export_df"] is None and "corrupt" in last["error"]


def _pump_gui(monkeypatch):
    shown, scheduled, states = [], [], []
    label = SimpleNamespace(text=None)
    label.configure = lambda text: setattr(label, "text", text)
    g = _gui(_busy=True, _exporting=True, progress={}, progress_bar_label=label,
             root=SimpleNamespace(after=lambda ms, fn: scheduled.append(fn)))
    g._update_button_states = lambda: states.append((g._busy, g._exporting))
    monkeypatch.setattr(sm.messagebox, "showerror", lambda title, msg, parent=None: shown.append(msg))
    return g, shown, scheduled, states, label


def test_done_with_error_unlocks_the_controls_and_shows_the_error(monkeypatch):
    g, shown, scheduled, states, label = _pump_gui(monkeypatch)
    g.progress_reporter.report(done=True, error="Analysis failed: boom")
    g._pump_progress()
    assert g._busy is False and states and shown == ["Analysis failed: boom"]
    assert "failed" in label.text and len(scheduled) == 1


def test_failed_export_unlocks_and_never_opens_the_save_dialog(monkeypatch):
    g, shown, scheduled, states, label = _pump_gui(monkeypatch)
    monkeypatch.setattr(gui_export, "asksaveasfilename", _boom)
    g.progress_reporter.report(export_done=True, export_df=None, error="Export failed: boom")
    g._pump_progress()
    assert g._busy is False and g._exporting is False and shown == ["Export failed: boom"]


def test_pump_is_rearmed_even_when_a_handler_raises(monkeypatch):
    g, shown, scheduled, states, label = _pump_gui(monkeypatch)
    monkeypatch.setattr(g, "_handle_progress_event", _boom)
    g.progress_reporter.report(value=10)
    try:
        g._pump_progress()
    except RuntimeError:
        pass
    assert len(scheduled) == 1


def test_export_write_error_is_shown_not_raised(monkeypatch):
    g, shown, scheduled, states, label = _pump_gui(monkeypatch)
    monkeypatch.setattr(gui_export, "asksaveasfilename", lambda **k: "Z:/nowhere/out.csv")
    df = SimpleNamespace(to_csv=lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked")))
    g._save_export(df)
    assert shown and "locked" in shown[0] and not hasattr(g, "last_export_path")


def test_workers_read_the_debug_snapshot_not_the_tk_variable():
    def tk_get():
        raise AssertionError("Tk variable read from a worker")

    g = _gui(debug_var=SimpleNamespace(get=tk_get))
    g._debug_snapshot = True
    assert g._debug_on() is True
    g2 = _gui(debug_var=SimpleNamespace(get=lambda: 1))
    g2._snapshot_tk_state()
    g2.debug_var = SimpleNamespace(get=tk_get)
    assert g2._debug_on() is True
