#!/usr/bin/env python3
"""Compose Actual R3 Demonstration Footage Into Presentation Videos."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry
from offroad_autonomy.runtime.video_recorder import FramePacer, build_ffmpeg_command
from offroad_autonomy.visualization.route_presentation import RoutePresentation


class Writer:
    def __init__(self, path, fps):
        if path.exists():
            raise FileExistsError(f"Preserve Existing Video: {path}")
        command = build_ffmpeg_command(
            shutil.which("ffmpeg"), path, 1920, 1080, fps, 20, "veryfast"
        )
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self.frames = 0

    def write(self, image):
        self.process.stdin.write(memoryview(np.ascontiguousarray(image)).cast("B"))
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        code = self.process.wait()
        if code:
            raise RuntimeError(f"Encoding Failed: {error}")


class Recording:
    def __init__(self, directory, geometry, route):
        summary = json.loads((directory / "run_01.summary.json").read_text())
        if not (
            summary["route_completed"]
            and summary["zero_manual_inputs_verified"]
            and summary["zero_authored_corridor_departures"]
            and summary["speed_compliance"]
        ):
            raise ValueError(f"Demonstration Did Not Meet Its Route/Safety Conditions: {directory}")
        capture = json.loads((directory / "capture.json").read_text())
        if capture["encoder_failed"] or capture["capture_error"]:
            raise ValueError(f"Camera Recording Failed: {directory}")
        pacer = FramePacer(capture["fps"])
        self.states = []
        station = route["start_station_m"]
        for line in (directory / "frames.jsonl").read_text().splitlines():
            record = json.loads(line)
            measured = geometry.project(record["native_pose"]["position"], station, 12, 3)
            station = measured["route_station_m"]
            record["progress_m"] = station - route["start_station_m"]
            record["offset_m"] = measured["signed_lateral_offset_m"]
            self.states.extend([record] * pacer.repeats(record["host_timestamp"]))
        self.reader = cv2.VideoCapture(str(directory / "cameras.mp4"))
        assert (
            len(self.states)
            == capture["frames_written"]
            == int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.fps = capture["fps"]
        self.current = -1
        self.image = None
        progress = np.maximum.accumulate([s["progress_m"] for s in self.states])
        self.progress, self.indices = np.unique(progress, return_index=True)

    def at_progress(self, progress):
        index = int(round(np.interp(progress, self.progress, self.indices)))
        return self.at_frame(index), self.states[index]

    def at_frame(self, index):
        if index < self.current or index > self.current + 20:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.current = index - 1
        while self.current < index:
            ok, self.image = self.reader.read()
            if not ok:
                raise RuntimeError(f"Missing Encoded Camera Frame {index}")
            self.current += 1
        return self.image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    args = parser.parse_args()
    route = yaml.safe_load((ROOT / "configs/r3_approved_bends.yaml").read_text())
    geometry = RoadGeometry(
        json.loads(
            (ROOT / "results/r3/final_verification/approved_route_geometry.json").read_text()
        )["road"]["edges"]
    )
    baseline = json.loads(
        (ROOT / "results/r3/final_verification/batch_01/aggregate.json").read_text()
    )
    improved = json.loads(
        (ROOT / "results/r3/tracking_improvement/verification_01/aggregate.json").read_text()
    )
    renderer = RoutePresentation(geometry, route, baseline, improved)
    before = Recording(args.recordings / "baseline", geometry, route)
    after = Recording(args.recordings / "improved", geometry, route)
    comparison_path = args.recordings / "R3_Curve_Improvement.mp4"
    comparison = Writer(comparison_path, 24)
    intro = renderer.intro()
    cv2.imwrite(str(args.recordings / "R3_Video_Poster.png"), intro)
    for _ in range(5 * 24):
        comparison.write(intro)
    chapters = []
    for name, first, last in (
        ("Western Bend And Transition", 20.0, 150.0),
        ("Eastern Return Bend", 210.0, 300.0),
    ):
        frames = int(round((last - first) / 3.0 * 24))
        chapters.append(
            {
                "title": name,
                "video_start_s": comparison.frames / 24,
                "course_progress_start_m": first,
                "course_progress_finish_m": last,
                "playback": "3 m Of Route Progress Per Video Second; Not Real-Time",
            }
        )
        for progress in np.linspace(first, last, frames):
            a, sa = before.at_progress(progress)
            b, sb = after.at_progress(progress)
            comparison.write(renderer.comparison(a, b, sa, sb, progress, name))
    closing = renderer.results()
    for _ in range(8 * 24):
        comparison.write(closing)
    comparison.close()
    print(f"Comparison Video Written: {comparison_path}", flush=True)
    full_path = args.recordings / "R3_Approved_Route_Full_Run.mp4"
    full = Writer(full_path, after.fps)
    for index, state in enumerate(after.states):
        full.write(renderer.full(after.at_frame(index), state))
        if index % 500 == 0:
            print(f"Full Route Rendering: {index}/{len(after.states)} Frames", flush=True)
    full.close()
    (args.recordings / "video_manifest.json").write_text(
        json.dumps(
            {
                "comparison": str(comparison_path),
                "comparison_frames": comparison.frames,
                "comparison_fps": 24,
                "full_route": str(full_path),
                "full_route_frames": full.frames,
                "full_route_fps": after.fps,
                "chapters": chapters,
                "source": "Actual Fresh BeamNG Demonstrations; No Generated Driving Footage",
                "statistics": "Separate Preserved Original And Improved Five-Run Verification Batches",
                "telemetry": "Latest Native Control Poll; Not Exposure-Synchronized",
                "scope": "Presentation Only; These Recordings Are Not Additional Verification Trials",
            },
            indent=2,
        )
    )
    print(f"Full Route Video Written: {full_path}", flush=True)


if __name__ == "__main__":
    main()
