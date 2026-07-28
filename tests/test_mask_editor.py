import numpy as np

from utils.mask_editor import MaskEditor


def _blank(size=100):
    return np.zeros((size, size), dtype=np.uint8)


def test_stroke_is_continuous_between_widely_spaced_positions():
    """A quick drag only produces a motion event every few tens of pixels; the
    pixels in between must still be painted."""
    editor = MaskEditor(_blank(), brush_radius=2)

    editor.begin_stroke()
    editor.paint(10, 50, add=True)
    editor.paint(90, 50, add=True)  # 80 px jump, as a fast drag would give
    editor.end_stroke()

    mask = editor.get_mask()
    # Every pixel along the path, not just the two endpoints.
    assert np.all(mask[50, 10:91] == 255)


def test_erase_leaves_no_trace_along_a_fast_drag():
    """The reported symptom: in erase mode the skipped pixels stayed behind as
    an unerased trace following the cursor."""
    editor = MaskEditor(np.full((100, 100), 255, dtype=np.uint8), brush_radius=3)

    editor.begin_stroke()
    editor.paint(10, 40, add=False)
    editor.paint(60, 40, add=False)
    editor.paint(60, 90, add=False)
    editor.end_stroke()

    mask = editor.get_mask()
    assert np.all(mask[40, 10:61] == 0)
    assert np.all(mask[40:91, 60] == 0)


def test_positions_are_not_joined_across_separate_strokes():
    """Lifting the button and pressing again elsewhere must not draw a line
    between the two places."""
    editor = MaskEditor(_blank(), brush_radius=2)

    editor.begin_stroke()
    editor.paint(10, 50, add=True)
    editor.end_stroke()
    editor.begin_stroke()
    editor.paint(90, 50, add=True)
    editor.end_stroke()

    mask = editor.get_mask()
    assert mask[50, 10] == 255
    assert mask[50, 90] == 255
    assert mask[50, 50] == 0  # midpoint untouched


def test_undo_restores_the_mask_from_before_the_stroke():
    editor = MaskEditor(_blank(), brush_radius=2)

    editor.begin_stroke()
    editor.paint(10, 50, add=True)
    editor.paint(90, 50, add=True)
    editor.end_stroke()
    assert editor.undo() is True

    assert not editor.get_mask().any()
    assert editor.undo() is False  # nothing left to undo
