# RGB Perception And Planning Audit

## Outcome

The implemented changes improve confidence handling, path validation, calibration support, timestamp reporting, and CPU postprocessing cost. They preserve YOLO26 and the existing simulation controller. **The segmentation accuracy target has not been achieved.** On all 1,245 locally available ORFD validation RGB images, the current checkpoint scores 0.4913 binary mIoU and 0.2735 road IoU. The shipped defaults produce exactly the same segmentation confusion matrix before and after the changes.

No controller implementation, controller tuning, model architecture, or dependencies were changed. Existing user changes, including the selected `models/weights.pt`, were retained. No commit was created. The initial working source/configuration snapshot is saved locally as `output/rgb_audit/working_baseline.zip`, with an extracted copy in `output/rgb_audit/working_baseline/`.

## Existing Pipeline And Safeguards

The production path is a single RGB camera, decoded once to OpenCV BGR, resized to 720x465, optionally enhanced with luminance CLAHE, and letterboxed by Ultralytics to the configured inference size (1024). The current checkpoint is `yolo26s-sem.yaml` with background and traversable-road classes. YOLOE-26 instance checkpoints also remain supported.

Already implemented before this work:

- Calibrated mount height/orientation and FOV-derived intrinsics shared by perception, planning, and display; normalized hood exclusion applied after inference and morphology.
- Optional CLAHE, close/open morphology, binary EMA, minimum mask area, per-pixel semantic probabilities, and instance-confidence weighting outside the hood.
- A confidence/area gate selecting the largest road component anchored near the visible vehicle edge. Distant disconnected blobs cannot start a path.
- A baseline planner following row runs, fitting a quadratic in ground coordinates, and blending at equal forward distances. Its existing blend checks approximately 90% waypoint support, but did not validate the full unsmoothed fit or the segments between waypoints.
- An advanced image-space planner, and an experimental BEV grid with wheel/gyro dead-reckoned motion. Grid clearance and candidate trajectory checks already exist.
- Reduced-speed simulation path holding followed by an empty path when perception fails, bounded timing windows, an asynchronous dashboard, cached text/static graphics, and explicitly counted superseded display snapshots.
- RGB-only BeamNG cameras with annotation, instance-annotation, and depth rendering disabled. A separate presentation camera is display-only.

The source tree contains older MPC/MPPI code and configurations. They predate this work and were not developed or tuned here. The default simulation controller remains Stanley.

## Findings And Implemented Changes

| Finding | Change | Expected Benefit And Limit |
| --- | --- | --- |
| A confident distant region could raise the whole-frame score while the selected corridor was weak | Retain an aligned confidence map for semantic and instance outputs. Filter current evidence before connectivity, then measure the selected component's confidence | Weak or non-finite pixels cannot connect a trusted corridor. Confidence still needs calibration against real labels |
| EMA or morphology could create planning support absent from the current prediction | Intersect planning evidence with the current raw mask and confidence map | Temporal history cannot fabricate fresh road support; raw/postprocessed masks remain available for evaluation |
| Grid integration previously consumed the full mask after the gate accepted one component | Integrate only trusted evidence connected to the chosen corridor, including its continuation beyond the gate ROI | Reject disconnected and weak evidence without discarding the visible road above the gate ROI |
| Quadratic fits and temporal filters could leave the road between samples | Check the complete polyline at half-pixel spacing after planning/smoothing, plus image boundary clearance | Reject unsupported curves and holes; this is conservative when segmentation is fragmented |
| A centerline alone does not establish vehicle-width clearance | Add an optional swept-width polygon check using measured half-width and a configurable margin, normal to the projected path | Reject too-narrow roads. The check is opt-in because mount, footprint, and ground assumptions must be validated; unsupported image-edge sections can cause rejection |
| Metric projection becomes unstable near the horizon | Reject baseline samples and baseline/advanced output points outside configurable range and pitch-sensitivity bounds | Expose sensitivity to a specified pitch envelope. This does not estimate slopes, crests, suspension motion, or actual attitude |
| Real cameras need independent focal lengths and principal points | Accept measured native-resolution `[fx, fy, cx, cy]`; independently scale both axes, including nonuniform resize | Operate on rectified real RGB without assuming centered principal points or equal scaled focal lengths |
| Offline/real-camera callers had to construct a simulation controller | Add `RGBPlanningPipeline.plan_result()` and `PlanningStepResult`; retain `AutonomyPipeline` as the simulation wrapper | Recorded RGB and ROS callbacks can receive trajectories/status without constructing a controller or connecting to BeamNG |
| Timing began after capture and preprocessing overwrote capture time | Preserve source timestamps; report perception/planning time, generation time, frame age, and timestamp provenance. Reject stale, future, negative, or non-finite known timestamps | Consumers can distinguish fresh, held, unavailable, and stale plans. Receive timestamps only bound age from receipt onward |
| Disabled EMA still allocated/copied float history | Bypass float history when EMA is off; update enabled EMA history in place | Measured 46% lower CPU postprocessing time with identical output masks in an interleaved test |
| Health displays used a 7 FPS / 143 ms target | Set dashboard health targets to 20 FPS / 50 ms | Make missed engineering targets visible; changing a target does not improve throughput |
| TensorRT export unconditionally attempted to prompt fixed semantic models | Preserve fixed semantic classes during export; retain YOLOE prompting | Fix an export setup error. Semantic TensorRT runtime routing and numerical equivalence remain unvalidated |

All held plans now have `valid=False`, `status="held"`; missing paths have `status="unavailable"`; timestamp failures have `status="stale"`. A validated centerline without a configured width has `status="centerline_only"` and `vehicle_width_validated=False`. Disabling corridor validation yields `status="unchecked"`. Experimental grid output is marked `grid_estimate`. `valid` describes current planning evidence, not a guarantee of physical collision avoidance. A physical consumer must inspect status, timestamp provenance, age, and width validation as well.

`planning.path_half_width_m` defaults to zero to preserve the working baseline. Set it to measured vehicle half-width, and set `path_boundary_margin_m` after validation. Width checking concerns a lateral corridor under the flat-ground assumption; it does not model wheel dynamics, swept vehicle length, overhang, or hidden obstacles. Existing controller clearance fields are not repurposed to change actuation behavior.

## Measured RGB Results

Hardware: RTX 3060 Laptop GPU (6 GiB), Windows, Python 3.11.14, PyTorch 2.5.1+CUDA. Both full runs use the same `weights.pt`, 1024 inference input, native label grid, and all 1,245 validation frames in filename/timestamp order. SHA-256 input manifests match. Five warm-up inferences are excluded from stage timings. Annotations are read only by the scorer, never supplied to the segmenter or planner. Dataset depth, LiDAR, and simulator-derived reference files are excluded.

| Full ORFD Validation Metric | Before | After |
| --- | ---: | ---: |
| Binary macro IoU (road/background) | 0.49130 | 0.49130 |
| Road IoU | 0.27348 | 0.27348 |
| Unwarped temporal road-mask IoU | 0.89054 | 0.89054 |
| Label-relative change agreement | 0.93920 | 0.93920 |
| Postprocessing mean / p95 | 1.780 / 2.181 ms | 0.958 / 1.134 ms |
| Preprocessing mean | 4.058 ms | 4.031 ms |
| Segmentation wall time mean / p95 | 33.578 / 60.496 ms | 30.474 / 35.325 ms |
| PNG decoding mean | 13.204 ms | 13.140 ms |
| Decode-through-mask mean / p95 | 52.621 / 80.624 ms | 48.604 / 54.646 ms |
| Processing capacity, excluding disk reads and scoring | 19.00 FPS | 20.57 FPS |
| Entire offline evaluation, including disk I/O and scoring | 10.89 FPS | 13.19 FPS |
| Average process CPU core equivalents | 1.98 | 2.09 |
| Peak allocated CUDA tensor memory | 131.76 MiB | 131.76 MiB |

These full runs have **planning disabled**, because the ORFD camera's ground-relative mounting calibration has not been established. The report key `rgb_to_path` consequently measures decode-through-mask here. These numbers do not establish live camera-to-trajectory latency, UI FPS, or ingestion rate. Do not add independent stage p95 values to estimate an end-to-end p95.

The GPU model was not optimized between these runs; GPU clocks, warm-up, power state, and scheduling affect the inference differences. Attribute the postprocessing gain to the isolated interleaved test, not all observed end-to-end variation. In that 300-sample comparison on identical annotation-shaped masks, postprocessing mean/p95 fell from **1.551/1.870 ms to 0.831/0.970 ms**, with exact mask and stability-score equality. Disabled float history removes several frame-sized allocations/copies. Model-to-CPU transfers, working-image resize, Ultralytics letterboxing, and GPU inference remain potential bottlenecks.

During 120 one-second machine-level telemetry samples overlapping the full comparisons, GPU utilization averaged 29.8% (0-69%); memory-controller utilization averaged 10.5%, and reported GPU memory use peaked at 315 MiB. This includes I/O/scoring gaps and is not isolated model utilization. CUDA allocated memory excludes reserved/context/other-process memory. CPU and GPU utilization are not interchangeable with throughput.

Off-screen rendering of 100 completed RGB snapshots measured **12.86 ms mean / 14.87 ms p95**. The benchmark opens no window; desktop window presentation and refresh rate remain unmeasured. Dashboard text caching and asynchronous rendering already exist, so no UI redesign was warranted.

### Preprocessing And Existing Checkpoint Experiments

Each ablation uses the same first 200 validation frames. These are validation experiments, not a separate held-out test result.

| Variant | mIoU | Road IoU | Temporal IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| Existing defaults, 1024 | 0.46471 | 0.24596 | 0.92365 | Retain |
| CLAHE | 0.39198 | 0.13384 | 0.83964 | Degrades accuracy and adds preprocessing cost |
| Close/open morphology | 0.46486 | 0.24620 | 0.92407 | Improvement too small to justify changing the default |
| EMA, alpha 0.7 | 0.46471 | 0.24596 | 0.92365 | Identical masks with extra cost |
| 640 inference input | 0.43448 | 0.18929 | 0.88653 | Faster inference but loses accuracy |
| Aspect-preserving 720x405 working image | 0.46533 | 0.24620 | 0.92601 | Tiny accuracy gain; retain camera-specific configuration |
| Native 1280x720 working image | 0.46704 | 0.24823 | 0.91973 | Small gain, still far below the accuracy target |
| Existing nano checkpoint | 0.33338 | 0.00000 | Undefined | No road predicted on this sample; do not substitute |
| Existing alternate small checkpoint | 0.33338 | 0.00000 | Undefined | No road predicted on this sample; do not substitute |

With binary input and fixed EMA alpha greater than 0.5, thresholding at 0.5 reproduces the current raw binary mask: a current positive contributes more than 0.5 and a current negative retains less than 0.5. This explains why alpha 0.7 did not improve accuracy or temporal stability. Lower alpha can introduce lag and ghost support; the current-evidence gate prevents that support from certifying a new corridor.

The default 720x465 working grid matches the BeamNG capture aspect, but resizes ORFD's 1280x720 RGB nonuniformly. The two additional working-size experiments above isolate this issue. Their accuracy gains are small. Their latency samples overlapped regression tests and are not used for performance comparisons; rerun them alone for throughput measurements. `--working-size 720 405` preserves the ORFD aspect, while `--working-size 1280 720` bypasses the first resize. The physical camera's calibration must match whichever input/output grids are used.

No tested preprocessing or existing alternate checkpoint closes the accuracy gap. Training provenance and annotation-definition compatibility have not been verified, so these results do not establish training/test independence or the cause of the generalization failure. The next accuracy work needs an agreed traversability-label definition, representative manually annotated deployment-camera sequences, and checkpoint validation or YOLO26 fine-tuning against those labels. Shadows, dirt/vegetation texture, lighting, and fragmented edges should be stratified in that set. No architecture change is justified by these measurements.

### Metric Definitions

- Spatial mIoU is the mean of aggregate foreground and background IoU across valid native-resolution annotation pixels. Road IoU is reported separately. ORFD value 255 means road, 0 means background, 128 is excluded; unmapped values fail rather than being guessed.
- tIoU is mean consecutive predicted-road-mask IoU in common valid pixels, without motion compensation. Gaps longer than 0.25 seconds reset history; 1,225 temporal pairs enter the full-run score. Empty predicted unions are undefined and excluded. Camera motion can reduce this metric despite correct segmentation.
- Frame stability is explicitly operationalized as agreement between predicted and annotated binary changes, restricted to the active road region. This is not a standardized motion-compensated stability score. A consistently wrong or empty mask can still score highly; compare it only alongside spatial accuracy and coverage.
- Recorded-RGB planning mode rasterizes the whole output polyline onto native manual annotations to measure road support, unknown coverage, and image clearance. Consecutive valid paths also report unregistered lateral target change at 5 m. This is a variability proxy, not motion-compensated trajectory error.

## Measured Planner Stress Results

These are **synthetic mask tests**, not real-world segmentation or trajectory-accuracy claims. Both source trees process the same 210 deterministic masks and confidence maps: straight, curved, narrow, S-bend, hole, weak corridor plus confident disconnected blob, and dropout sequences. Input hashes match. No reference centerline is supplied to the planner.

| Metric | Before | After, Default Centerline Validation |
| --- | ---: | ---: |
| Plans accepted as current | 204/210 | 144/210 |
| Accepted paths crossing non-road pixels between waypoints | 30 | 0 |
| Accepted weak-confidence corridors | 30 | 0 |
| Minimum image clearance among accepted paths | 0.00 px | 4.20 px |
| Mean consecutive target change at fixed image row | 0.559 px | 0.554 px |
| Planning mean / p95 | 3.031 / 3.738 ms | 4.552 / 5.977 ms |

Availability decreased deliberately: the hole and weak-corridor cases now fail. Straight, curved, S-bend, and narrow-centerline cases retain 30/30 accepted paths each; the six dropout frames are invalid. Acceptance sets differ, so the aggregate target-change statistic is not a causal comparison of smoothing. There is no substantial demonstrated smoothing/stability gain; the measurable improvement is rejecting unsupported trajectories.

With a configured 0.95 m half-width, the strict test accepts straight and broad-curved roads, rejects all 30 narrow-road cases and all S-bend cases whose width/visibility cannot certify the footprint, and accepts 84/210 frames overall. Its planning mean/p95 was **5.971/8.712 ms**. This conservative tradeoff is why width checking is configurable. Full physical trajectory stability and metric boundary clearance require recorded RGB plus measured calibration and independent reference annotations; the ORFD camera-to-ground extrinsics are not guessed from the simulator or LiDAR.

## Reproducing The Measurements

Run from the repository root in the existing `offroad-seg` environment:

The initial RGB audit below used FP32. With the later FPS optimization applied,
add `--no-half` to these RGB benchmark commands to reproduce that precision.
The current default and FP16 comparisons are documented in
[Measured FPS Improvements](fps_optimization.md).

```powershell
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --out output/rgb_audit/replay.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --clahe --out output/rgb_audit/replay_clahe.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --morphology --out output/rgb_audit/replay_morphology.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --ema --out output/rgb_audit/replay_ema.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 200 --input-size 640 --out output/rgb_audit/replay_640.json
python scripts/benchmark_rgb.py --images datasets/ORFD/validation/image_data --labels datasets/ORFD/validation/gt_image --limit 100 --render-ui --out output/rgb_audit/replay_ui.json
python scripts/benchmark_planning.py --out output/rgb_audit/planning_replay.json
python scripts/benchmark_planning.py --half-width-m 0.95 --out output/rgb_audit/planning_width_replay.json
python scripts/benchmark_postprocess.py --baseline-root output/rgb_audit/working_baseline --labels datasets/ORFD/validation/gt_image --out output/rgb_audit/postprocess_replay.json
```

For the saved baseline, set `PYTHONPATH` to the extracted snapshot's `src`, and run its saved `scripts/benchmark_rgb.py` with its saved config. Run `benchmark_planning.py` from the current scripts against the same baseline `PYTHONPATH`. Restore the original `PYTHONPATH` afterward. Keep the model weights fixed; compare input hashes and checkpoint hashes. The baseline snapshot includes the evaluation code used for its full run. Newer evaluation reports also record checkpoint SHA-256.

Use `--threshold` for confidence sweeps and `--weights` to compare existing checkpoints. Filenames must be numeric timestamps; set `--timestamp-scale` to their seconds conversion and `--sequence-gap-s` to the permitted sequence gap. The default naming/mapping matches local ORFD data. Different annotation palettes need an explicit conversion to indexed grayscale, not grayscale guesses for color labels.

For an independently annotated deployment-camera sequence with established calibration, add `--planning --config <camera-config.yaml>`. Set the correct sensor resolution, native intrinsics, mount pose, hood mask, processing resolution, and measured vehicle width first. Do not use the BeamNG mount for unrelated real footage. Manual labels must correspond to the same rectified frames. For live simulation profiling, use the existing `offroad-autonomy --benchmark-seconds 30 --benchmark-out output/rgb_audit/live.json` after starting/configuring BeamNG. Its report now includes changed-RGB versus repeated-poll counts and frame-age stage statistics.

## Real-Camera And ROS Integration

`RGBPlanningPipeline` accepts an OpenCV BGR array or `CameraFrame`. The default baseline planner requires no vehicle state. It does not connect to BeamNG or construct a controller:

```python
from offroad_autonomy.pipeline import RGBPlanningPipeline
from offroad_autonomy.types import CameraFrame
from offroad_autonomy.utils.config import load_config

pipeline = RGBPlanningPipeline(load_config("camera.yaml"))
# exposure_monotonic must be mapped to time.perf_counter()'s clock domain.
result = pipeline.plan_result(CameraFrame(
    image=rectified_bgr,
    timestamp=exposure_monotonic,
    frame_id=exposure_id,
    timestamp_source="capture",
))
# Publish the trajectory together with plan.valid, plan.status,
# vehicle_width_validated, frame_age_ms and the source timestamp.
```

A top-level `camera` YAML section can override the historical `beamng.camera` settings for real-camera use. `camera.intrinsics` contains `[fx, fy, cx, cy]` at the configured native sensor resolution. Input must already be rectified: lens distortion and rolling-shutter compensation are not implemented. Camera height must be measured relative to the assumed ground plane. Coordinate conventions remain +X left, -Y forward, +Z up; a physical adapter must convert its measured mount into that convention.

No ROS 2 node, message definitions, launch configuration, or installed ROS graph was found in this checkout. The NumPy/dataclass boundary avoids new ROS dependencies. Actual ROS conversion, transport, executor contention, QoS, and synchronization overhead remain unmeasured; preserving an external ROS adapter's ABI cannot be verified here. Publish validity/status rather than trajectories alone, and map sensor/ROS clocks explicitly before passing timestamps.

## Ingestion, Latency, And Sim-To-Real Limitations

1. **28 FPS ingestion is not established.** Requesting a 30 Hz BeamNG sensor does not prove the host ingests 28 unique exposures. The current host loop polls synchronously before inference. Source frames produced between polls are unobservable, and a content signature cannot distinguish identical consecutive exposures from a frozen stream. New benchmark counters explicitly describe this limitation. Every frame delivered to the pipeline is still processed; no new latest-frame inference queue or frame-dropping policy was added.
2. If every 28+ FPS exposure needs a mask, sustained processing must also reach 28+ FPS. A 20 FPS consumer and a 28 FPS lossless producer cannot maintain both a bounded queue and bounded latency. A physical camera adapter needs exposure IDs/timestamps, accepted/dropped-frame counters, and a clearly agreed backlog policy. This engineering requirement remains open rather than being silently relaxed.
3. BeamNG only supplies a host receive timestamp here. `timestamp_source="host_receive"` means reported age excludes unknown exposure/transport delay. The code does not age a static scene merely because its pixels repeat. Real capture timestamps need clock mapping; timestamp zero means unknown and reports age as null. `runtime.max_frame_age_ms=200` is a stale-output guard, distinct from the 50 ms performance goal, and can be disabled with zero for deliberate offline replay.
4. RGB alone does not establish absolute terrain shape, hidden obstacles, friction, or a safe corridor over a crest. Range and pitch-sensitivity limits reject ill-conditioned rays but cannot certify the flat-ground assumption. No depth, road geometry, semantic annotations, GPS, or perfect pose is used to hide this limitation.
5. The experimental grid uses signed wheel speed and yaw-rate integration, with real equivalents in wheel encoders and a gyro. The simulator supplies ideal readings rather than realistic sensor noise/slip. Its local pose is dead reckoning, not simulator global pose; physical covariance and synchronization remain to be validated. Grid mode does not yet apply the new per-ray baseline projection sensitivity check and retains its documented range/flat-ground limitations.
6. Old `deprecated/` implementations and historical `output/` reference-route/stereo reports are not the deployed pipeline and were not used as evaluation truth. Map/spawn values in simulation setup only initialize test scenarios. Road/terrain lookup APIs are absent from the active perception/planning path.
7. Jetson and TensorRT hardware/runtime are unavailable. FP16 export must be built on the target GPU and compared against the `.pt` reference on the same labeled sequences. Semantic engine task selection and confidence-predictor routing need explicit validation: the current loader's exported-model task defaults to instance segmentation. Successful semantic export is not proof of supported semantic-engine deployment. No claimed Jetson speedup or INT8 accuracy result is supplied.
8. A live BeamNG process was not running during the initial RGB audit. Subsequent live dashboard and FP16 benchmarks are recorded in [Measured FPS Improvements](fps_optimization.md). Those short live sessions validate integration and measure timing but do not establish physical safety or trajectory accuracy. No physical actuation or vehicle-dynamics optimization was attempted.

## Changed Files And Validation

Core changes: `types.py`, `pipeline.py`, `utils/config.py`, `perception/road_segmenter.py`, `perception/camera_geometry.py`, `planning/perception_gate.py`, `planning/baseline_planner.py`, `planning/centerline_planner.py`, new `planning/corridor_validation.py`, and `postprocessing/temporal_stabilizer.py`.

Runtime/integration changes: `runtime/timing.py`, `runtime/benchmark.py`, new `runtime/rgb_evaluation.py`, `simulation/beamng_client.py` (timestamp provenance only), and `main.py` (pass capture metadata to the existing recorder). Configuration/documentation: `configs/default.yaml`, `README.md`, and this audit. Reproduction tools: new `scripts/benchmark_rgb.py`, `scripts/benchmark_planning.py`, `scripts/benchmark_postprocess.py`, and the fixed `scripts/export_engine.py`.

Tests: new `test_rgb_reliability.py`, `test_rgb_benchmark.py`, and `test_export_engine.py`; extended `test_semantic_segmentation.py`, `test_baseline_planner.py`, `test_dashcam_capture.py`, `test_pipeline.py`, `test_runtime.py`, `test_dashboard.py`, and `test_presentation.py`. Tests cover independent label metrics, annotation isolation, replay hood exclusion, component-local confidence, disconnected grid evidence, missing/low/non-finite predictions, full-polyline gaps, width checks, calibrated intrinsics, projection uncertainty, stale/future timestamps, processing versus changed-frame rate, updated health targets, simulator-independent planning, and export class handling.

Validation: Ruff lint and format checks passed; full regression/coverage run passed **392 tests, 1 skipped, 89.61% coverage**. Package sdist and wheel built successfully into `output/rgb_audit/dist/` using `python -m build --no-isolation --outdir output/rgb_audit/dist`. The first attempt to overwrite existing `dist/` artifacts was blocked by file permissions; the separate output directory succeeded. Benchmark JSON/telemetry/source snapshots are local ignored artifacts under `output/rgb_audit/`. The generated coverage database is saved there as well; the tracked baseline coverage file is restored to avoid an unrelated binary change.

The 0.85 mIoU target is unmet. Temporal proxies exceed the stated thresholds on this validation set but do not prove accurate segmentation. A 50 ms camera-to-plan p95, sustained 20 FPS live UI/perception, and 28 FPS ingestion remain unverified. Further acceptance requires calibrated, manually labeled RGB deployment sequences and live target-hardware measurements, not simulator ground truth.
