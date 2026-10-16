# Measured FPS Improvements

## Findings And Changes

The installed OpenCV build has no GUI support, so the application selects
Tkinter. The existing asynchronous display worker already owned window creation
and destruction, but `main.py` forced every Tkinter window onto the main loop.
Rendering and displaying the dashboard added about 51 ms after each primary loop.

The changes preserve YOLO26, its weights, 1024-pixel inference input, the 720x465
working mask, the camera, the planner's safety checks, and Stanley behavior:

- Enable asynchronous Tkinter display on Windows when Tcl reports thread support.
  Create, draw, and destroy the interpreter and its images on the same worker.
  Retain inline rendering on unsupported Tkinter platforms. Protect held-key state
  when it is read across threads.
- Reuse the Tk photo until the window size changes. Pass its interpreter explicitly
  and release all Tcl-backed references on the owning thread during shutdown.
- Replace full-canvas NumPy padding with OpenCV border copying, producing identical
  pixels. Skip resizing when source and display dimensions match.
- Honor the configured display rate even when new snapshots arrive during its
  wait. Superseded display snapshots remain counted; no inference frames are
  discarded by this change.
- Add configurable `perception.half`, enabled in the shipped config after a labeled
  FP32/FP16 comparison. CUDA uses FP16 network inference; CPU falls back to FP32.
  Semantic logit interpolation and probability calculations still use FP32.
  This does not replace the architecture or reduce image resolution.
- Include dashboard rate, render/show timings, failures, and superseded snapshot
  counts in live benchmark JSON reports.

## Live Measurements

Measured on an RTX 3060 Laptop GPU with BeamNG running on the same machine,
`automation_test_track`, spawn 0, default dashcam and live dashboard. Baseline and
final sessions ran for 20 seconds, with three seconds excluded from the recorder's
per-frame metrics. Short sessions do not establish sustained thermal performance.

| Metric | Original Inline FP32 | Async FP32 | Final Async FP16 |
| --- | ---: | ---: | ---: |
| Main loop FPS | 8.14 | 12.60 | 13.76 |
| Full loop mean | 122.25 ms | 78.18 ms | 71.66 ms |
| Full loop p95 | 147.92 ms | 88.78 ms | 100.07 ms |
| Primary loop mean | 70.80 ms | 77.46 ms | 70.97 ms |
| Primary loop p95 | 84.13 ms | 88.03 ms | 99.29 ms |
| Perception + planning mean | 64.10 ms | 66.58 ms | 42.12 ms |
| Perception + planning p95 | 67.04 ms | 71.05 ms | 58.60 ms |
| Host-receive-to-plan mean | 69.89 ms | 73.75 ms | 63.86 ms |
| Dashboard FPS | Not separately saved | 12.65 | 13.98 |

Exact values are in `output/fps_optimization/before_live.json`,
`async_fp32_live.json`, and `final_live.json`. The source/config baseline is retained
in `output/fps_optimization/before_source.zip`. The intermediate
`async_fp16_live.json` session ended early and is not the final comparison.

The final main-loop rate increased about 69%; mean full-loop time fell about 41%.
The primary-loop p95 worsened. It would be incorrect to claim that all latency
metrics improved or that the 20 FPS / 50 ms targets were achieved. Per-stage
statistics use rolling windows that include startup, while recorder-level
primary/full-loop metrics exclude warmup, so their means need not match.

The final run spent another 20.21 ms on average reading wheel/gyro state via a
synchronous simulator RPC and 7.86 ms sending commands. These are outside the
RGB perception/planning algorithm. The primary loop also experienced simulator
and dashboard CPU/GPU contention. No control changes or simulator physics tuning
were made to improve these numbers. Live sessions have the same map and spawn,
but do not contain exactly identical camera exposures; timing comparisons are
indicative, not controlled trajectory-accuracy comparisons.

The final dashboard reported zero draw failures and zero superseded snapshots.
All received RGB frames still go through inference. The final run observed
13.57 changed RGB polls per second, not certified exposure ingestion. A requested
30 Hz camera does not demonstrate the required 28 unique exposures per second.
Host receive timestamps also omit unknown exposure and transport delay.

## RGB Accuracy And Isolated Timing

Both precisions were evaluated on the same first 200 ORFD validation RGB frames
and manual labels, using the same weights, threshold, and spatial grids. Input
and weights hashes are recorded in `fp32_rgb.json` and `fp16_rgb.json`.

| Metric | FP32 | FP16 |
| --- | ---: | ---: |
| mIoU | 0.464708 | 0.464778 |
| Road IoU | 0.245956 | 0.246056 |
| Unwarped temporal IoU | 0.923648 | 0.923666 |
| Label-relative change agreement | 0.957333 | 0.957332 |
| Segmentation mean, no simulator | 17.12 ms | 15.23 ms |
| Segmentation p95, no simulator | 23.85 ms | 36.00 ms |

Masks are not identical between precisions. This small check found negligible
aggregate accuracy change; it does not certify other scenes or hardware. The
0.85 mIoU target remains unmet and needs better matched labeled training/evaluation
data. These isolated runs have planning disabled because the ORFD camera's mounting
calibration is not established. They cannot establish live FPS or camera-to-path
latency. The worse FP16 p95 also shows why mean-only claims are insufficient.

The interleaved CPU display-padding benchmark measured 7.887 to 2.331 ms mean and
8.455 to 2.675 ms p95 at 1896x990, with identical pixels. This component result is
not itself a UI FPS improvement measurement.

## Reproduction And Configuration

From the repository root in the project environment:

```powershell
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --no-half --out output/fps_optimization/repeat_fp32.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --half --out output/fps_optimization/repeat_fp16.json
python scripts/benchmark_display_padding.py --out output/fps_optimization/repeat_padding.json
offroad-autonomy --benchmark-seconds 20 --benchmark-warmup 3 --benchmark-out output/fps_optimization/repeat_live.json --log-level INFO
```

Run these sequentially without simultaneous training, tests, or other GPU jobs.
For a precision A/B in BeamNG, use a config overlay with `perception.half: false`.
For an inline-display comparison, additionally set `ui.display_async: false`.
An exact original-code comparison requires the saved baseline source snapshot;
an inline run on the new code still contains the new padding implementation.

The real Tk lifecycle regression can be run explicitly:

```powershell
$env:OFFROAD_TEST_TK_GUI='1'
python -m pytest -q tests/test_window_thread.py
```

It briefly opens a real window and verifies worker-owned creation, repeated
drawing, and destruction. Normal headless tests skip this opt-in GUI check.
Jetson/TensorRT and real-camera timing remain unmeasured. No dependency was added,
and no simulator road geometry, pose, GPS, depth, or semantic ground truth was
introduced into perception or planning.

## Files Changed For This Optimization

`configs/default.yaml`, `types.py`, `utils/config.py`, and
`perception/road_segmenter.py` add the precision setting. `main.py`,
`runtime/display_worker.py`, and `visualization/window.py` change dashboard
threading, pacing, reporting, and allocation. `scripts/benchmark_rgb.py` adds
precision overrides; `scripts/benchmark_display_padding.py` supplies the isolated
CPU benchmark. The config, display worker, semantic segmentation, and window test
modules cover the changed behavior. README and both audit documents describe the
results and limitations. Earlier RGB reliability changes remain separately
documented in [RGB Pipeline Audit](rgb_pipeline_audit.md).

The full regression suite passed: 405 passed, two GUI-dependent tests skipped,
90.34% coverage. The explicit real Windows Tk worker test also passed in the
focused GUI run. Ruff lint and formatting checks passed. Wheel/sdist builds and
the existing offline S-bend regression passed without changing the controller.
