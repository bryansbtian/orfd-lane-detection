"""Dashboard video recording: pacing, encoder settings and never blocking."""

import shutil
import threading
import time

import numpy as np
import pytest

from offroad_autonomy.runtime import video_recorder
from offroad_autonomy.runtime.video_recorder import FramePacer, VideoRecorder, build_ffmpeg_command


def test_frames_faster_than_the_video_rate_are_skipped():
    pacer = FramePacer(fps=10.0)

    assert pacer.repeats(0.00) == 1
    assert pacer.repeats(0.02) == 0
    assert pacer.repeats(0.10) == 1


def test_a_stalled_dashboard_repeats_frames_so_playback_keeps_real_time():
    pacer = FramePacer(fps=10.0)

    pacer.repeats(0.0)

    # 0.5 s without a frame fills slots 1 to 5.
    assert pacer.repeats(0.5) == 5


def test_ffmpeg_is_asked_for_h264_that_browsers_and_vs_code_can_play(tmp_path):
    command = build_ffmpeg_command(
        "ffmpeg", tmp_path / "run.mp4", 1600, 900, 20.0, crf=23, preset="veryfast"
    )
    joined = " ".join(command)

    assert "-c:v libx264" in joined
    assert "-pix_fmt yuv420p" in joined
    assert "-movflags +faststart" in joined
    assert "-pix_fmt bgr24 -s 1600x900 -r 20 -i -" in joined


def test_odd_dimensions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="even"):
        VideoRecorder(tmp_path / "run.mp4", 1601, 900, fps=20.0)


def test_missing_ffmpeg_fails_with_an_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(video_recorder.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        VideoRecorder(tmp_path / "run.mp4", 4, 4, fps=20.0)


class _StalledPipe:
    """Stands in for an encoder that cannot keep up."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.writes = 0

    def write(self, data) -> None:
        self.release.wait()
        self.writes += 1

    def close(self) -> None:
        self.release.set()


class _FakeProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.stdin = _StalledPipe()
        self.stderr = None
        self.returncode = 0

    def wait(self, timeout=None) -> int:
        return 0

    def poll(self) -> int:
        return 0


def test_a_slow_encoder_drops_frames_instead_of_blocking_the_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(video_recorder.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_recorder.subprocess, "Popen", _FakeProcess)
    recorder = VideoRecorder(tmp_path / "run.mp4", 4, 4, fps=20.0, queue_frames=2)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    for i in range(20):
        recorder.write(frame, i / 20.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5
    assert recorder.frames_dropped >= 16

    recorder._process.stdin.release.set()
    recorder.close(timeout=2.0)
    assert recorder.frames_written + recorder.frames_dropped >= 20


def test_a_frame_of_the_wrong_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(video_recorder.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_recorder.subprocess, "Popen", _FakeProcess)
    recorder = VideoRecorder(tmp_path / "run.mp4", 4, 4, fps=20.0)

    with pytest.raises(ValueError):
        recorder.write(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)

    recorder.close(timeout=2.0)


_FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg is not installed")
@pytest.mark.skipif(
    _FFMPEG is not None and _FFMPEG.startswith("/snap/"),
    reason="snap confinement gives ffmpeg a private /tmp, so it cannot write tmp_path",
)
def test_recording_produces_a_playable_h264_mp4(tmp_path):
    path = tmp_path / "run.mp4"
    # This test submits ten frames without pacing; leave room for the whole
    # burst so encoder startup timing cannot drop a frame from the test clip.
    recorder = VideoRecorder(path, 64, 48, fps=20.0, preset="ultrafast", queue_frames=10)
    rng = np.random.default_rng(0)

    for i in range(10):
        recorder.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), i / 20.0)
    recorder.close()

    data = path.read_bytes()
    assert not recorder.failed
    assert recorder.frames_written == 10
    assert b"avc1" in data
    assert data.find(b"moov") < data.find(b"mdat")
