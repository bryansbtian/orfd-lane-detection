# Off-Road Autonomy

If you are new to the repo, read these first:

1. `src/offroad_autonomy/main.py` for the runtime loop, BeamNG connection, and
   dashboard updates.
2. `src/offroad_autonomy/pipeline.py` for the module order and the handoff
   between stages.
3. `configs/default.yaml` for the map, spawn, camera, model, controller, and
   visualization settings.

## Pipeline At A Glance

```text
BeamNG Frame -> Preprocessing -> Perception -> Postprocessing -> Planning -> Control -> BeamNG Vehicle
```

Visualization runs alongside that loop and overlays the postprocessed mask, the
planned path, and runtime telemetry on top of the front camera view.

## The Seven Modules

| Module         | Where it lives                         | High-level responsibility                                                                                                                                                                     |
| -------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BeamNG Client  | `src/offroad_autonomy/simulation/`     | Starts the simulator, loads the map and spawn point, attaches the front camera, reads frames and telemetry, and sends controls back to BeamNG.                                                |
| Preprocessing  | `src/offroad_autonomy/preprocessing/`  | Resizes incoming frames from `1280x720` to `640x360` and can apply CLAHE to improve contrast before segmentation.                                                                             |
| Perception     | `src/offroad_autonomy/perception/`     | Runs the configured YOLO-based segmentation model, uses open-vocabulary prompts or trained classes to detect traversable terrain, and merges instance masks into one binary traversable mask. |
| Postprocessing | `src/offroad_autonomy/postprocessing/` | Smooths the raw mask over time with EMA and morphology so the road region is more stable from frame to frame.                                                                                 |
| Planning       | `src/offroad_autonomy/planning/`       | Extracts a centerline, estimates heading and road width, and uses a Kalman filter plus fallback logic to keep the path usable when perception becomes noisy.                                  |
| Control        | `src/offroad_autonomy/control/`        | Converts the planned path and current vehicle state into steering, throttle, and brake using Stanley steering and proportional speed control.                                                 |
| Visualization  | `src/offroad_autonomy/visualization/`  | Renders the development dashboard with the front camera view, traversable mask, planned path, and metrics like confidence, stability, FPS, and latency.                                       |

## One Loop Iteration

1. The BeamNG client captures a new RGB frame and the latest vehicle state.
2. Preprocessing normalizes the image so downstream modules see a consistent
   input size and contrast.
3. Perception produces a traversable-road mask plus model confidence and
   inference timing.
4. Postprocessing stabilizes that mask so small flicker and noise do not
   immediately disturb planning.
5. Planning turns the stabilized mask into a centerline reference path and can
   fall back to Kalman prediction if the mask becomes unreliable.
6. Control converts the path into a `ControlCommand` for steering, throttle,
   and brake.
7. Visualization overlays the latest outputs so we can debug the behavior
   during closed-loop runs.

## Why The Split Matters

The module boundaries are intentional. They make it easier to tune one part of
the system without rewriting the others, and they let us debug failures by
asking a simple question: did the issue start in the image, the mask, the path,
or the controller output?

## Where To Go Next

- If you want to change simulator setup, maps, spawns, or sensors, start in
  `src/offroad_autonomy/simulation/` and `configs/default.yaml`.
- If you want to improve road detection, start in
  `src/offroad_autonomy/perception/`.
- If you want to improve path stability, look at
  `src/offroad_autonomy/postprocessing/` and
  `src/offroad_autonomy/planning/`.
- If you want to tune steering or speed behavior, look at
  `src/offroad_autonomy/control/`.
- If you want to change the operator view, start in
  `src/offroad_autonomy/visualization/`.

## Running Locally

```bash
pip install -e ".[dev]"
```

Set `beamng.home` in `configs/default.yaml` to your local BeamNG.tech install,
then run:

```bash
offroad-autonomy --config configs/default.yaml --log-level INFO
```

You can also use the helper script:

```bash
python scripts/run_main.py --config configs/default.yaml
```

Run the test suite with:

```bash
pytest -v
```

## Repository Layout

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
|   |-- utils/
|   `-- visualization/
|-- tests/
|-- datasets/
|-- models/
```
