"""Read Presentation Cameras Without Changing The R3 Driving Or Safety Stack."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from offroad_autonomy.runtime.video_recorder import VideoRecorder
from offroad_autonomy.simulation.r3_verification_client import R3VerificationClient


class PresentationCaptureClient(R3VerificationClient):
    def __init__(self, config, directory: Path) -> None:
        super().__init__(config)
        if not self._shared_memory:
            raise RuntimeError("Presentation Capture Requires Local Shared-Memory Cameras")
        self._orbit = True
        self._directory = directory
        self._latest_pose = None
        self._latest_dashcam = None
        self._command = {}
        self._lock = threading.Lock()
        self._stop_capture = threading.Event()
        self._capture_thread = None
        self._video = None
        self._capture_error = None
        self._recorded = 0

    def capture_frame(self):
        frame = super().capture_frame()
        if frame is not None:
            # The presentation receives the exact inference snapshot, never a
            # second dashcam sample with an unrelated perception timestamp.
            with self._lock:
                self._latest_dashcam = frame.image
        return frame

    def native_status(self):
        status = super().native_status()
        with self._lock:
            self._latest_pose = status.get("latest")
        return status

    def send_controls(self, command):
        super().send_controls(command)
        with self._lock:
            self._command = {
                "steering": command.steering,
                "throttle": command.throttle,
                "brake": command.brake,
            }

    def start_physics_recording(self, path, margin):
        result = super().start_physics_recording(path, margin)
        self._directory.mkdir(exist_ok=True)
        self._video = VideoRecorder(
            self._directory / "cameras.mp4", 1920, 728, 20, crf=19, preset="veryfast"
        )
        self._capture_thread = threading.Thread(target=self._capture, daemon=True)
        self._capture_thread.start()
        return result

    def _capture(self):
        try:
            with (self._directory / "frames.jsonl").open("x", encoding="utf-8") as stream:
                while not self._stop_capture.is_set():
                    started = time.perf_counter()
                    with self._lock:
                        pose = self._latest_pose
                        dashcam = self._latest_dashcam
                        command = self._command
                    if pose is not None and dashcam is not None:
                        # Only mapped-memory reads run here. BeamNG socket/Lua
                        # requests stay on the controller thread because their
                        # connection is not thread safe.
                        raw = bytes(self._cameras["orbit"].colour_shmem.read())
                        orbit = self._decode(raw, self._config.orbit_camera)
                        if orbit is not None:
                            canvas = np.zeros((728, 1920, 3), dtype=np.uint8)
                            canvas[:, :960] = orbit
                            canvas[54:674, 960:] = dashcam
                            dropped = self._video.frames_dropped
                            self._video.write(canvas, started)
                            if self._video.frames_dropped == dropped:
                                stream.write(
                                    json.dumps(
                                        {
                                            "host_timestamp": started,
                                            "native_pose": pose,
                                            "command": command,
                                        }
                                    )
                                    + "\n"
                                )
                                stream.flush()
                                self._recorded += 1
                    remaining = max(0, 0.05 - (time.perf_counter() - started))
                    self._stop_capture.wait(remaining)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._capture_error = f"{type(exc).__name__}: {exc}"

    def _finish_capture(self):
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
            if self._capture_thread.is_alive():
                raise RuntimeError("Presentation Capture Did Not Stop")
            self._capture_thread = None
        if self._video is not None:
            self._video.close()
            result = {
                "frames_enqueued": self._recorded,
                "frames_written": self._video.frames_written,
                "frames_dropped": self._video.frames_dropped,
                "encoder_failed": self._video.failed,
                "capture_error": self._capture_error,
                "fps": 20,
                "canvas_size": [1920, 728],
                "left_view": "Actual Chase Camera, 960x728",
                "right_view": "Inference Dashcam Snapshot, 960x620, Letterboxed",
                "scope": "Fresh Demonstration; Excluded From Frozen Verification Results",
                "telemetry_alignment": "Most Recent Native Control Poll; Not Exposure-Synchronized",
            }
            (self._directory / "capture.json").write_text(json.dumps(result, indent=2))
            self._video = None

    def stop_physics_recording(self):
        self._finish_capture()
        return super().stop_physics_recording()

    def disconnect(self):
        self._finish_capture()
        super().disconnect()
