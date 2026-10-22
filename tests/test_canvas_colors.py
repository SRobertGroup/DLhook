"""The zoomable canvas shows a cv2 (BGR) image with true colours (audit M-3)."""
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


def test_bgr_image_is_displayed_as_rgb(root, monkeypatch):
    import ui.zoomable_canvas as zc

    shown = []
    real = zc.ImageTk.PhotoImage
    monkeypatch.setattr(zc.ImageTk, "PhotoImage", lambda image: shown.append(np.asarray(image)) or real(image=image))
    canvas = zc.ZoomableImageCanvas(root, width=100, height=100)
    image = np.zeros((20, 20, 3), np.uint8)
    image[..., 0] = 255                                  # pure blue in BGR
    canvas.set_image(image, reset_view=True)
    centre = shown[-1][shown[-1].shape[0] // 2, shown[-1].shape[1] // 2]
    assert tuple(centre) == (0, 0, 255)                  # blue in RGB
    assert canvas._image[0, 0, 0] == 255                 # the stored image is left as it was
    canvas.destroy()
