"""The dashboard window lives and dies on one thread.

Win32 delivers a window's messages only to the thread that created it. The
window used to be created on the main thread and drawn from the display
worker, so it never repainted, Windows marked it Not Responding, and the
worker itself could block for good. These tests pin the lifecycle: open,
show and close all on the worker thread, close in the worker's ``finally``,
and never a window on any other thread.
"""

import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker
from offroad_autonomy.visualization import window as window_module
from offroad_autonomy.visualization.window import DashboardWindow


class _Recorder:
    """Stand-in window that records which thread did what."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def open(self) -> None:
        self.calls.append(("open", threading.get_ident()))

    def show(self, frame) -> bool:
        self.calls.append(("show", threading.get_ident()))
        return True

    def close(self) -> None:
        self.calls.append(("close", threading.get_ident()))

    def threads(self, action: str) -> set[int]:
        return {ident for name, ident in self.calls if name == action}


def _worker(recorder: _Recorder, asynchronous: bool, **kwargs) -> DisplayWorker:
    return DisplayWorker(
        render=lambda state: np.zeros((4, 4, 3), dtype=np.uint8),
        show=recorder.show,
        read_key=lambda: -1,
        rate_hz=0.0,
        asynchronous=asynchronous,
        open_window=kwargs.pop("open_window", recorder.open),
        close_window=recorder.close,
        **kwargs,
    )


def _state() -> DisplayState:
    return DisplayState(result=object(), telemetry=object())


def test_async_worker_opens_draws_and_closes_on_its_own_thread():
    recorder = _Recorder()
    worker = _worker(recorder, asynchronous=True)

    worker.start()
    deadline = time.monotonic() + 5.0
    while worker.rendered < 3 and time.monotonic() < deadline:
        worker.publish(_state())
        time.sleep(0.01)
    worker.stop()

    assert worker.rendered >= 3
    owner = recorder.threads("open")
    assert len(owner) == 1
    assert recorder.threads("show") == owner
    assert recorder.threads("close") == owner
    assert threading.get_ident() not in owner
    names = [name for name, _ in recorder.calls]
    assert names[0] == "open"
    assert names[-1] == "close"


def test_async_worker_closes_the_window_even_if_drawing_ends_it():
    """The operator closing the window ends the loop; the finally block must
    still destroy it on the worker thread."""
    recorder = _Recorder()
    worker = _worker(recorder, asynchronous=True)
    worker._show = lambda frame: False  # the window reports itself closed

    worker.start()
    worker.publish(_state())
    deadline = time.monotonic() + 5.0
    while not worker.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert worker.closed
    assert len(recorder.threads("close")) == 1
    assert recorder.threads("close") == recorder.threads("open")


def test_inline_worker_opens_and_closes_on_the_calling_thread():
    recorder = _Recorder()
    worker = _worker(recorder, asynchronous=False)

    worker.start()
    worker.publish(_state())
    worker.stop()

    me = {threading.get_ident()}
    assert recorder.threads("open") == me
    assert recorder.threads("show") == me
    assert recorder.threads("close") == me


def test_window_that_cannot_open_disables_the_dashboard_not_the_run():
    recorder = _Recorder()

    def refuse() -> None:
        raise RuntimeError("no display")

    worker = _worker(recorder, asynchronous=True, open_window=refuse)
    worker.start()
    deadline = time.monotonic() + 5.0
    while not worker.unavailable and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.publish(_state())
    worker.stop()

    assert worker.unavailable
    # "closed" is what ends the run; a missing window must not.
    assert not worker.closed
    assert worker.rendered == 0
    assert recorder.threads("close") == set()


# --- DashboardWindow itself ----------------------------------------------------


@pytest.fixture
def fake_highgui(monkeypatch):
    """OpenCV's window calls, recorded instead of performed."""
    created: list[int] = []
    destroyed: list[int] = []
    cv2 = window_module.cv2
    monkeypatch.setattr(window_module, "_opencv_gui_available", lambda: True)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: created.append(threading.get_ident()))
    monkeypatch.setattr(cv2, "resizeWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: destroyed.append(threading.get_ident()))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return created, destroyed


def test_constructing_the_window_creates_no_os_window(fake_highgui):
    created, _ = fake_highgui

    window = DashboardWindow("test", 64, 48)

    assert window.backend == "opencv"
    assert not window.is_open
    assert created == []


def test_close_off_the_owning_thread_is_refused(fake_highgui):
    _, destroyed = fake_highgui
    window = DashboardWindow("test", 64, 48)
    opener = threading.Thread(target=window.open)
    opener.start()
    opener.join()

    with pytest.raises(RuntimeError, match="owning thread"):
        window.close()
    assert destroyed == []


def test_show_before_open_is_refused(fake_highgui):
    window = DashboardWindow("test", 64, 48)

    with pytest.raises(RuntimeError, match="before open"):
        window.show(np.zeros((4, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("threaded", [False, True])
def test_tk_async_requires_supported_threaded_runtime(monkeypatch, threaded):
    monkeypatch.setattr(window_module, "_opencv_gui_available", lambda: False)
    monkeypatch.setattr(window_module, "_tkinter_available", lambda: True)
    monkeypatch.setattr(window_module, "_threaded_tk_available", lambda: threaded)
    window = DashboardWindow("test", 64, 48)
    assert window.supports_async is threaded


@pytest.mark.parametrize("size", [(1600, 900), (1896, 990), (200, 400), (1, 1), (401, 199)])
def test_display_padding_matches_reference_pixels(size):
    cv2 = window_module.cv2
    frame = np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    tw, th = size
    scale = min(tw / 160, th / 90)
    nw, nh = max(1, round(160 * scale)), max(1, round(90 * scale))
    reference = np.full((th, tw, 3), (17, 22, 26), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    reference[oy : oy + nh, ox : ox + nw] = cv2.resize(frame, (nw, nh))
    actual = DashboardWindow._prepare_frame_for_display(frame, tw, th, (17, 22, 26))
    np.testing.assert_array_equal(actual, reference)


def test_tk_reuses_photo_until_resize_and_releases_references_on_close(monkeypatch):
    monkeypatch.setattr(window_module, "_opencv_gui_available", lambda: False)
    monkeypatch.setattr(window_module, "_tkinter_available", lambda: True)
    window = DashboardWindow("test", 64, 48)
    root, label, image_tk = Mock(), Mock(), Mock()
    label.winfo_width.return_value = 64
    label.winfo_height.return_value = 48
    photo = Mock()
    photo.width.return_value = 64
    photo.height.return_value = 48
    image_tk.PhotoImage.return_value = photo
    window._root, window._label, window._image_tk = root, label, image_tk
    window._tk = SimpleNamespace(TclError=RuntimeError)
    window._owner = threading.get_ident()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert window.show(frame)
    assert window.show(frame)
    image_tk.PhotoImage.assert_called_once()
    assert image_tk.PhotoImage.call_args.kwargs["master"] is root
    photo.paste.assert_called_once()
    label.winfo_width.return_value = 80
    assert window.show(frame)
    assert image_tk.PhotoImage.call_count == 2
    window._on_key_press(SimpleNamespace(keysym="W"))
    assert window.held_keys == {"w"}
    window._on_key_release(SimpleNamespace(keysym="W"))
    assert window.held_keys == set()
    window.close()
    root.destroy.assert_called_once()
    assert window._photo is None
    assert window._label is None
    assert window._root is None
    assert window._tk is None
    assert window._image_tk is None


def test_real_tk_window_round_trip_on_worker_thread(monkeypatch):
    if os.environ.get("OFFROAD_TEST_TK_GUI") != "1":
        pytest.skip("set OFFROAD_TEST_TK_GUI=1 for real GUI lifecycle validation")
    if not window_module._threaded_tk_available():
        pytest.skip("needs Windows threaded Tcl")
    monkeypatch.setattr(window_module, "_opencv_gui_available", lambda: False)
    window = DashboardWindow("orfd Tk worker test", 160, 90)
    worker = DisplayWorker(
        render=lambda state: np.full((90, 160, 3), 40, dtype=np.uint8),
        show=window.show,
        read_key=lambda: window.last_key,
        rate_hz=0,
        asynchronous=window.supports_async,
        open_window=window.open,
        close_window=window.close,
    )
    worker.start()
    deadline = time.monotonic() + 10
    while worker.rendered < 20 and time.monotonic() < deadline:
        worker.publish(_state())
        time.sleep(0.01)
    worker.stop(timeout=5)
    assert worker.rendered >= 20
    assert worker.failed == 0
    assert worker._thread is None
    assert not window.is_open
    assert window._root is None
    assert window._photo is None


def test_real_window_round_trip_on_the_worker_thread():
    """The regression itself, on a real Win32 window: the worker must keep
    drawing and then stop. The old cross-thread window blocked it for good."""
    if sys.platform != "win32" or not window_module._opencv_gui_available():
        pytest.skip("needs a Win32 HighGUI build")
    window = DashboardWindow("orfd window thread test", 160, 90)
    worker = DisplayWorker(
        render=lambda state: np.full((90, 160, 3), 40, dtype=np.uint8),
        show=window.show,
        read_key=lambda: window.last_key,
        rate_hz=0.0,
        asynchronous=True,
        open_window=window.open,
        close_window=window.close,
    )

    worker.start()
    deadline = time.monotonic() + 10.0
    while worker.rendered < 20 and time.monotonic() < deadline:
        worker.publish(_state())
        time.sleep(0.01)
    worker.stop(timeout=5.0)

    assert worker.rendered >= 20
    assert worker._thread is None
    assert not window.is_open
