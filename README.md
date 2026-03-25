# offroad_autonomy

End-to-end autonomous off-road driving for [BeamNG.tech](https://www.beamng.tech/).

The stack takes a front camera stream from BeamNG, segments the traversable
region, turns that mask into a centerline, and converts the centerline into
steering, throttle, and brake commands.

## Pipeline Overview

The runtime loop is:

```text
BeamNG camera -> preprocessing -> perception -> postprocessing -> planning -> control -> BeamNG vehicle
```

At a high level:

1. `main.py` loads configuration, starts logging, connects to BeamNG, and owns
   the outer loop.
2. `BeamNGClient` captures the latest camera frame and vehicle telemetry.
3. `AutonomyPipeline.step()` runs the frame through every processing stage.
4. The resulting `ControlCommand` is sent back to the simulator.

This split is intentional. The application loop is responsible for lifecycle,
signals, and simulator I/O. The pipeline is responsible only for converting a
frame plus vehicle state into a control output. That separation keeps the core
autonomy logic testable and lets individual stages be swapped without rewriting
the rest of the system.

### Stage Summary

| Stage | Input | Output | Why it exists |
| --- | --- | --- | --- |
| `simulation` | BeamNG session | Camera frames and telemetry | Keeps all `beamngpy` details in one place |
| `preprocessing` | Raw BGR frame | `FramePacket` | Standardizes image size and optionally improves contrast |
| `perception` | `FramePacket` | `PerceptionResult` | Converts pixels into a binary traversable-road mask |
| `postprocessing` | `PerceptionResult` | `StabilizedResult` | Reduces frame-to-frame jitter and removes small noise |
| `planning` | `StabilizedResult` | `PathPlan` | Extracts a drivable centerline and heading estimate |
| `control` | `PathPlan`, `VehicleState` | `ControlCommand` | Turns the plan into steering, throttle, and brake |

## Module Walkthrough

### `src/offroad_autonomy/main.py`

Process: parses CLI arguments, loads `configs/default.yaml`, configures logging,
builds the simulator client and autonomy pipeline, then runs a loop that
captures a frame, polls vehicle state, computes a command, and sends controls.
It also installs a signal handler so the loop exits cleanly on `Ctrl+C`.

Why: `main.py` is the operational shell around the autonomy stack. Keeping
startup, shutdown, argument parsing, and long-running loop behavior here keeps
the rest of the code focused on autonomy logic instead of process management.

### `src/offroad_autonomy/pipeline.py`

Process: creates one instance of each stage and exposes a single `step()`
method. For every frame, it runs:

1. `ImagePreprocessor.process()`
2. `RoadSegmenter.predict()`
3. `TemporalStabilizer.stabilize()`
4. `CenterlinePlanner.plan()`
5. `StanleyController.compute()`

It also exposes `reset()` for clearing temporal state when the scene changes.

Why: the pipeline is deliberately thin. It should orchestrate the stages, not
hide their behavior. That makes the control flow easy to read and makes unit
tests straightforward because each stage can be mocked independently.

### `src/offroad_autonomy/simulation/beamng_client.py`

Process: launches BeamNG, loads the configured map, spawns the ego vehicle,
attaches a front camera, waits for the simulator and sensor stream to settle,
captures color frames, polls vehicle state, and sends actuator commands back to
the vehicle. It also resolves spawn points from the YAML map table and converts
quaternions into Euler yaw for the controller pipeline.

Why: BeamNG-specific concerns are noisy and failure-prone compared with the
rest of the stack. Isolating them behind `BeamNGClient` means the autonomy
pipeline does not need to know anything about scenario loading, sensors, or
`beamngpy`. The short startup waits are there to avoid racing the physics world
or camera stream before both are ready.

### `src/offroad_autonomy/preprocessing/image_preprocessor.py`

Process: takes the raw BGR image from BeamNG, records a timestamp, resizes the
frame to the configured processing resolution, and optionally applies CLAHE in
LAB color space. The module returns a `FramePacket` that contains both the
original image and the processed one.

Why: downstream modules need a stable input size. Without that, mask geometry,
path extraction, and controller assumptions all change with camera resolution.
CLAHE is optional because off-road scenes often have harsh lighting, shadows,
and washed-out dirt textures; contrast enhancement can help segmentation, but it
should remain configurable because it can also amplify noise.

### `src/offroad_autonomy/perception/road_segmenter.py`

Process: loads a YOLO/YOLOE segmentation model, optionally sets open-vocabulary
text prompts from configuration, runs inference on the preprocessed image, and
merges all predicted instance masks into a single binary traversable-road mask.
If mask dimensions do not match the frame size exactly, they are resized before
being combined. The module also retains per-instance confidence values and
inference time for diagnostics.

Why: the planner does not need object classes or separate instances; it needs a
single answer to "where can the vehicle drive?" Combining instance masks into
one boolean mask simplifies every downstream stage. Supporting prompt-based
open-vocabulary models and fine-tuned checkpoints through the same interface
makes the perception backend replaceable without changing the rest of the
pipeline.

### `src/offroad_autonomy/postprocessing/temporal_stabilizer.py`

Process: converts the binary mask to float, blends it with an exponential moving
average accumulator, thresholds the result back to a binary mask, applies
morphological close/open cleanup, rejects masks below a minimum area threshold,
and computes a temporal IoU score against the previous stabilized mask.

Why: raw segmentation is often noisy frame-to-frame, especially on dirt,
vegetation edges, and shadows. EMA smoothing reduces flicker, morphology closes
small holes and removes isolated specks, and the minimum-area filter prevents
tiny false positives from being treated as a road. The IoU stability score is a
simple health signal for debugging and future monitoring.

### `src/offroad_autonomy/planning/centerline_planner.py`

Process: checks whether the stabilized mask contains enough road pixels to be
usable. If it does, the planner samples several horizontal rows, finds the left
and right road edges in each row, and uses the midpoint as the centerline. From
the resulting points it estimates heading and average road width, then updates a
Kalman tracker over lateral offset, heading, and curvature. If the mask is too
small or the centerline is incomplete, the planner falls back to Kalman
prediction and emits a straight-ahead path based on the tracked state.

Why: for this stack, a centerline is the simplest path representation that is
cheap to compute from a segmentation mask. Row-wise midpoint extraction works
well when the main question is "stay near the middle of the traversable area."
The Kalman filter exists to preserve continuity when perception briefly fails;
without it, the controller would abruptly lose a path and produce unstable
behavior.

### `src/offroad_autonomy/control/stanley_controller.py`

Process: computes steering from the planned heading plus a Stanley cross-track
term based on the bottom-most centerline point in image space. It separately
computes throttle and brake from a proportional speed controller around a target
speed, then clips the final steering command to BeamNG's `[-1, 1]` range.

Why: Stanley control is a pragmatic fit here because the planner already
produces a centerline and heading-like signal, and the controller only needs a
lightweight geometric correction term rather than a full vehicle model. The
longitudinal controller is intentionally simple because this project is focused
on lateral path following over off-road terrain, not high-fidelity speed
planning.

### `src/offroad_autonomy/types.py`

Process: defines the data contracts shared between stages:
`FramePacket`, `PerceptionResult`, `StabilizedResult`, `PathPlan`,
`VehicleState`, `ControlCommand`, and `PipelineConfig`.

Why: these dataclasses keep the module boundaries explicit. A stage depends on a
stable typed interface rather than another stage's internal implementation.
That makes refactoring safer and reduces tight coupling between modules.

### `src/offroad_autonomy/utils/config.py` and `configs/default.yaml`

Process: `load_config()` reads the YAML file, splits it into domain sections
such as `beamng`, `perception`, `planning`, and `control`, and flattens those
settings into a single `PipelineConfig` object. The YAML currently holds map and
spawn data, camera settings, segmentation prompts, postprocessing thresholds,
planner tuning, and controller gains.

Why: centralizing tunable parameters in YAML keeps experiments out of the code.
Changing maps, camera placement, model weights, prompts, or control gains should
not require editing Python files. The flat `PipelineConfig` dataclass gives the
runtime code simple attribute access while preserving a human-friendly YAML file.

### `src/offroad_autonomy/utils/logger.py`

Process: creates a single console logger with a consistent format and guards
against adding duplicate handlers.

Why: long-running autonomy loops produce a lot of diagnostic output. A single,
predictable logger setup makes debugging easier and avoids repeated handler
registration when the application is initialized more than once.

## Configuration

All major tuning lives in `configs/default.yaml`. The most important groups are:

- `beamng`: simulator path, host/port, map, spawn index, camera settings
- `perception`: model weights, confidence threshold, prompt list
- `preprocessing`: resize dimensions and optional CLAHE settings
- `postprocessing`: EMA and morphology cleanup settings
- `planning`: centerline sampling, Kalman noise, fallback behavior
- `control`: Stanley gain, speed target, throttle/brake limits

Example:

```yaml
perception:
  model_weights: "models/yoloe-26x-seg.pt"
  confidence_threshold: 0.25
  prompts:
    - "traversable road"
    - "dirt road"
    - "off-road trail"
```

## Quick Start

```bash
# Install the package and dev tools
pip install -e ".[dev]"

# Make sure beamng.home in configs/default.yaml points to your BeamNG.tech install

# Run the autonomy stack
offroad-autonomy --config configs/default.yaml --log-level INFO

# Equivalent convenience script
python scripts/run_main.py --config configs/default.yaml

# Run tests
pytest tests/ -v
```

## Project Layout

```text
.
|-- configs/
|   `-- default.yaml
|-- scripts/
|   `-- run_main.py
|-- src/offroad_autonomy/
|   |-- main.py
|   |-- pipeline.py
|   |-- types.py
|   |-- control/
|   |-- perception/
|   |-- planning/
|   |-- postprocessing/
|   |-- preprocessing/
|   |-- simulation/
|   `-- utils/
|-- tests/
|-- datasets/
|-- models/
`-- deprecated/
```

## License

MIT
