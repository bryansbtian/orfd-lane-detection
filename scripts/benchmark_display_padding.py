"""Compare display padding costs without a simulator or GUI window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from offroad_autonomy.visualization.window import DashboardWindow


def _reference(frame, width, height, background):
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), background, np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations < 20:
        parser.error("--iterations must be at least 20")
    frame = np.random.default_rng(0).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
    background = (17, 22, 26)
    functions = {"before": _reference, "after": DashboardWindow._prepare_frame_for_display}
    samples = {key: [] for key in functions}
    np.testing.assert_array_equal(
        _reference(frame, 1896, 990, background),
        functions["after"](frame, 1896, 990, background),
    )
    # Alternate ordering so warming CPU caches does not always favour one version.
    for index in range(args.iterations + 10):
        order = list(functions)
        if index % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            functions[name](frame, 1896, 990, background)
            if index >= 10:
                samples[name].append((time.perf_counter() - start) * 1000)
    report = {
        "identical_pixels": True,
        "source_size": [1600, 900],
        "display_size": [1896, 990],
        "iterations": args.iterations,
        "opencv_threads": cv2.getNumThreads(),
        "stages_ms": {
            name: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}
            for name, values in samples.items()
        },
        "note": "CPU display preparation only; not UI FPS or perception throughput.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
