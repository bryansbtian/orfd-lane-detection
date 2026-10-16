"""Dashboard window backends for development visualization."""

from __future__ import annotations

import logging
import sys
import threading

import cv2

logger = logging.getLogger("offroad_autonomy.visualization")

_WINDOWS = sys.platform == "win32"
_VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44, "space": 0x20}


def _async_held_keys() -> set[str]:
    """HighGUI only reports key presses, not holds, so manual driving needs
    the OS key state; only Win32 exposes it without extra dependencies."""
    if not _WINDOWS:
        return set()
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return {name for name, vk in _VK.items() if user32.GetAsyncKeyState(vk) & 0x8000}
    except Exception:
        return set()


def _opencv_gui_available() -> bool:
    """Read from the build rather than by opening a probe window: a probe
    would belong to whichever thread made it, and only the drawing thread
    may own the dashboard window."""
    for line in cv2.getBuildInformation().splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            value = stripped.split(":", 1)[1].strip().upper()
            return value not in ("", "NONE")
    return False


def _tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401

        from PIL import ImageTk  # noqa: F401
    except Exception as exc:
        logger.warning("Tkinter window backend unavailable: %s", exc)
        return False
    return True


def _threaded_tk_available() -> bool:
    # Cocoa requires the main thread. Windows can own a threaded Tcl
    # interpreter entirely on the display worker, including its destruction.
    if not _WINDOWS:
        return False
    try:
        import tkinter as tk

        interpreter = tk.Tcl()
        return bool(int(interpreter.call("set", "tcl_platform(threaded)")))
    except Exception:
        return False


class DashboardWindow:
    """One OS window, owned by exactly one thread.

    Win32 delivers a window's messages only to the thread that created it,
    so a window created on one thread and pumped from another never repaints
    and Windows marks it Not Responding. Construction therefore only picks a
    backend. :meth:`open`, :meth:`show` and :meth:`close` must all run on the
    same thread - the display worker's - and :meth:`close` refuses to run
    anywhere else.
    """

    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.last_key: int = -1
        self._held: set[str] = set()
        self._held_lock = threading.Lock()
        self._backend = ""
        self._closed = False
        self._owner: int | None = None
        self._root = None
        self._label = None
        self._photo = None
        self._tk = None
        self._image_tk = None
        self._background = (17, 22, 26)

        if _opencv_gui_available():
            self._backend = "opencv"
        elif _tkinter_available():
            self._backend = "tkinter"
        else:
            raise RuntimeError("No supported GUI backend is available for the dashboard window.")
        logger.info("Dashboard window backend: %s", self._backend)

    @property
    def supports_async(self) -> bool:
        if self._backend == "opencv":
            return True
        return _threaded_tk_available()

    @property
    def backend(self) -> str:
        """The selected GUI backend; its lifecycle stays on its owner thread."""
        return self._backend

    @property
    def is_open(self) -> bool:
        return self._owner is not None

    @property
    def held_keys(self) -> set[str]:
        if self._backend == "opencv":
            return _async_held_keys()
        with self._held_lock:
            return set(self._held)

    def open(self) -> None:
        """Creates the window on the calling thread, which becomes its owner."""
        if self._owner is not None:
            raise RuntimeError("Dashboard window is already open")
        if self._backend == "opencv":
            self._open_opencv()
        else:
            self._open_tkinter()
        self._owner = threading.get_ident()

    def _open_opencv(self) -> None:
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_FREERATIO"):
            flags |= cv2.WINDOW_FREERATIO
        try:
            cv2.namedWindow(self.title, flags)
            cv2.resizeWindow(self.title, self.width, self.height)
        except cv2.error as exc:
            raise RuntimeError(f"OpenCV could not open the dashboard window: {exc}") from exc

    def _open_tkinter(self) -> None:
        import tkinter as tk

        from PIL import ImageTk

        try:
            root = tk.Tk()
            root.title(self.title)
            root.geometry(f"{self.width}x{self.height}")
            root.resizable(True, True)
            root.configure(bg="#11161a")
            root.protocol("WM_DELETE_WINDOW", self._request_close)
            root.bind("<Escape>", lambda event: self._request_close())
            root.bind("<KeyPress>", self._on_key_press)
            root.bind("<KeyRelease>", self._on_key_release)
            label = tk.Label(root, bd=0, bg="#11161a", highlightthickness=0)
            label.pack(fill="both", expand=True)
            root.update_idletasks()
        except Exception as exc:
            raise RuntimeError(f"Tkinter could not open the dashboard window: {exc}") from exc

        self._root = root
        self._label = label
        self._tk = tk
        self._image_tk = ImageTk

    def _on_key_press(self, event) -> None:
        sym = event.keysym.lower()
        if sym in ("w", "a", "s", "d"):
            with self._held_lock:
                self._held.add(sym)
        elif sym == "space":
            with self._held_lock:
                self._held.add("space")
        else:
            key_map = {"q": ord("q"), "e": ord("e"), "p": ord("p"), "t": ord("t"), "escape": 27}
            key_map.update({str(digit): ord(str(digit)) for digit in range(10)})
            if sym in key_map:
                code = key_map[sym]
                self.last_key = code
                if code in (27, ord("q")):
                    self._request_close()

    def _on_key_release(self, event) -> None:
        with self._held_lock:
            self._held.discard(event.keysym.lower())

    def show(self, frame_bgr) -> bool:
        """False once the window has been closed."""
        if self._closed:
            return False
        self._require_owner("show")
        self.last_key = -1

        if self._backend == "opencv":
            return self._show_opencv(frame_bgr)

        if self._backend == "tkinter":
            return self._show_tkinter(frame_bgr)

        return False

    def _show_opencv(self, frame_bgr) -> bool:
        target_w, target_h = self._opencv_viewport_size()
        display = self._prepare_frame_for_display(frame_bgr, target_w, target_h, self._background)
        cv2.imshow(self.title, display)
        key = cv2.waitKey(1) & 0xFF
        self.last_key = key
        if key in (27, ord("q")):
            self._closed = True
            return False

        try:
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                self._closed = True
                return False
        except cv2.error:
            self._closed = True
            return False
        return True

    def _show_tkinter(self, frame_bgr) -> bool:
        if self._closed or self._root is None or self._label is None:
            return False

        from PIL import Image

        target_w = max(int(self._label.winfo_width()), 1)
        target_h = max(int(self._label.winfo_height()), 1)
        if target_w <= 1 or target_h <= 1:
            target_w, target_h = self.width, self.height

        display = self._prepare_frame_for_display(frame_bgr, target_w, target_h, self._background)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if (
            self._photo is None
            or self._photo.width() != target_w
            or self._photo.height() != target_h
        ):
            self._photo = self._image_tk.PhotoImage(image=image, master=self._root)
            self._label.configure(image=self._photo)
        else:
            self._photo.paste(image)
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._closed = True
            return False

        return not self._closed

    def _opencv_viewport_size(self) -> tuple[int, int]:
        if hasattr(cv2, "getWindowImageRect"):
            try:
                _, _, width, height = cv2.getWindowImageRect(self.title)
                if width > 0 and height > 0:
                    return int(width), int(height)
            except cv2.error:
                pass
        return self.width, self.height

    @staticmethod
    def _prepare_frame_for_display(
        frame_bgr,
        target_w: int,
        target_h: int,
        background: tuple[int, int, int],
    ):
        src_h, src_w = frame_bgr.shape[:2]
        if src_h <= 0 or src_w <= 0:
            return frame_bgr

        target_w = max(int(target_w), 1)
        target_h = max(int(target_h), 1)
        if (src_w, src_h) == (target_w, target_h):
            return frame_bgr
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if new_w == target_w and new_h == target_h:
            return resized

        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        # OpenCV fills only the padding in native code; NumPy broadcasting
        # filled an entire large RGB canvas before overwriting its centre.
        return cv2.copyMakeBorder(
            resized,
            off_y,
            target_h - new_h - off_y,
            off_x,
            target_w - new_w - off_x,
            cv2.BORDER_CONSTANT,
            value=background,
        )

    def _request_close(self) -> None:
        self._closed = True

    def _require_owner(self, action: str) -> None:
        if self._owner is None:
            raise RuntimeError(f"Dashboard window {action}() before open()")
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                f"Dashboard window {action}() called off its owning thread; "
                "Win32 would leave the window unresponsive"
            )

    def close(self) -> None:
        """Destroys the window. Must run on the thread that opened it."""
        self._closed = True
        if self._owner is None:
            return
        self._require_owner("close")
        self._owner = None

        if self._backend == "opencv":
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            return

        if self._backend == "tkinter" and self._root is not None:
            # Tcl-backed objects must lose their final references on the
            # owner thread, before the worker exits and the main thread GCs.
            self._photo = None
            self._label = None
            try:
                self._root.destroy()
            except Exception as exc:
                logger.debug("Tkinter window was already gone: %s", exc)
            self._root = None
            self._tk = None
            self._image_tk = None
