"""Dashboard window backends for development visualization."""

from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logger = logging.getLogger("offroad_autonomy.visualization")

_WINDOWS = sys.platform == "win32"
_VK = {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44, "space": 0x20}


def _async_held_keys() -> set[str]:
    """Poll currently held drive keys via Win32 GetAsyncKeyState."""
    if not _WINDOWS:
        return set()
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return {name for name, vk in _VK.items() if user32.GetAsyncKeyState(vk) & 0x8000}
    except Exception:
        return set()


class DashboardWindow:
    """Display dashboard frames using OpenCV when possible, else Tkinter."""

    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.last_key: int = -1
        self._held: set[str] = set()
        self._backend = ""
        self._closed = False
        self._root = None
        self._label = None
        self._photo = None
        self._tk = None
        self._image_tk = None
        self._background = (17, 22, 26)

        if self._try_opencv():
            return
        if self._try_tkinter():
            return

        raise RuntimeError("No supported GUI backend is available for the dashboard window.")

    @property
    def held_keys(self) -> set[str]:
        if self._backend == "opencv":
            return _async_held_keys()
        return set(self._held)

    def _try_opencv(self) -> bool:
        try:
            flags = cv2.WINDOW_NORMAL
            if hasattr(cv2, "WINDOW_FREERATIO"):
                flags |= cv2.WINDOW_FREERATIO
            cv2.namedWindow(self.title, flags)
            cv2.resizeWindow(self.title, self.width, self.height)
        except cv2.error as exc:
            logger.warning("OpenCV window backend unavailable: %s", exc)
            return False

        self._backend = "opencv"
        logger.info("Dashboard window backend: OpenCV")
        return True

    def _try_tkinter(self) -> bool:
        try:
            import tkinter as tk
            from PIL import ImageTk
        except Exception as exc:
            logger.warning("Tkinter window backend unavailable: %s", exc)
            return False

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
            logger.warning("Failed to initialise Tkinter dashboard window: %s", exc)
            return False

        self._backend = "tkinter"
        self._root = root
        self._label = label
        self._tk = tk
        self._image_tk = ImageTk
        logger.info("Dashboard window backend: Tkinter")
        return True

    def _on_key_press(self, event) -> None:
        sym = event.keysym.lower()
        if sym in ("w", "a", "s", "d"):
            self._held.add(sym)
        elif sym == "space":
            self._held.add("space")
        else:
            key_map = {"q": ord("q"), "e": ord("e"), "p": ord("p"), "escape": 27}
            if sym in key_map:
                code = key_map[sym]
                self.last_key = code
                if code in (27, ord("q")):
                    self._request_close()

    def _on_key_release(self, event) -> None:
        self._held.discard(event.keysym.lower())
        if event.keysym.lower() == "space":
            self._held.discard("space")

    def show(self, frame_bgr) -> bool:
        """Display the latest dashboard frame. Returns False when closed."""
        if self._closed:
            return False
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
        self._photo = self._image_tk.PhotoImage(image=image)
        self._label.configure(image=self._photo)
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
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if new_w == target_w and new_h == target_h:
            return resized

        canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def _request_close(self) -> None:
        self._closed = True

    def close(self) -> None:
        """Close the active dashboard window backend safely."""
        self._closed = True

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
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
