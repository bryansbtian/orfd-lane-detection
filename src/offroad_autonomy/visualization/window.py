"""Dashboard window backends for development visualization."""

from __future__ import annotations

import logging

import cv2

logger = logging.getLogger("offroad_autonomy.visualization")


class DashboardWindow:
    """Display dashboard frames using OpenCV when possible, else Tkinter."""

    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.width = width
        self.height = height
        self._backend = ""
        self._closed = False
        self._root = None
        self._label = None
        self._photo = None
        self._tk = None
        self._image_tk = None

        if self._try_opencv():
            return
        if self._try_tkinter():
            return

        raise RuntimeError("No supported GUI backend is available for the dashboard window.")

    def _try_opencv(self) -> bool:
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
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
            from PIL import Image, ImageTk
        except Exception as exc:  # pragma: no cover - depends on local environment
            logger.warning("Tkinter window backend unavailable: %s", exc)
            return False

        try:
            root = tk.Tk()
            root.title(self.title)
            root.geometry(f"{self.width}x{self.height}")
            root.configure(bg="#11161a")
            root.protocol("WM_DELETE_WINDOW", self._request_close)
            root.bind("<Escape>", lambda event: self._request_close())
            root.bind("q", lambda event: self._request_close())
            label = tk.Label(root, bd=0, bg="#11161a", highlightthickness=0)
            label.pack(fill="both", expand=True)
            root.update_idletasks()
        except Exception as exc:  # pragma: no cover - depends on local environment
            logger.warning("Failed to initialise Tkinter dashboard window: %s", exc)
            return False

        self._backend = "tkinter"
        self._root = root
        self._label = label
        self._tk = tk
        self._image_tk = ImageTk
        logger.info("Dashboard window backend: Tkinter")
        return True

    def show(self, frame_bgr) -> int:
        """Display the latest dashboard frame.

        Returns
        -------
        int
            The key code pressed this frame, or -1 if none.
            Returns -2 when the window has been closed.
        """
        if self._closed:
            return -2

        if self._backend == "opencv":
            cv2.imshow(self.title, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._closed = True
                return -2

            try:
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    self._closed = True
                    return -2
            except cv2.error:
                self._closed = True
                return -2
            return key if key != 255 else -1

        if self._backend == "tkinter":
            return -1 if self._show_tkinter(frame_bgr) else -2

        return -2

    def _show_tkinter(self, frame_bgr) -> bool:
        if self._closed or self._root is None or self._label is None:
            return False

        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
