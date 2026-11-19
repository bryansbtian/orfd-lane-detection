# Off-Road Autonomy

## Introduction

RGB road segmentation and trajectory planning for unmarked trails, with BeamNG.tech
for simulation testing. YOLO26 predicts traversable road, a baseline planner fits the
trail's centreline, and a Stanley controller steers. Alternative planners and learned
controllers live in [Experiments](experiment/README.md) and never run by default.

Perception and planning use camera images and calibration, without simulator road
geometry, depth buffers, GPS, or perfect vehicle pose. Flat-ground projection and
segmentation errors remain deployment limitations.

## Development Setup

Requires Python 3.10+, BeamNG.tech 0.38 on Windows, and model weights at
`models/weights.pt`. An NVIDIA GPU is recommended for real-time inference.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate with `source .venv/bin/activate`. Semantic road class
names are selected by `perception.prompts` in [Default Config](configs/default.yaml).

## Run Locally

For automatic launch on the Windows simulator host:

```powershell
$env:BEAMNG_HOME = "<your BeamNG.tech folder>"
offroad-autonomy
```

The simulator connection uses port **64256**. To attach to an existing server,
set `BEAMNG_LAUNCH=false`. For a remote Windows simulator host, also set
`BEAMNG_HOST` to its address and allow that port through the host firewall.
Use `BEAMNG_PORT` if the server listens on a different port. Keep these local
environment settings out of committed configs.

Dashboard keys: `E` safe stop, `P` resume, `0/1/6/9` debug views, `T` timing, `Q` quit.
Use `--headless` to run without a window or `--config <file>` for a config overlay.

## Default Run

`offroad-autonomy` with [Default Config](configs/default.yaml) runs one pipeline:

1. The GMSL2 dashcam frame is resized to 720x465 and segmented by `perception.model_weights`.
2. The perception gate in `planning.gate` accepts the frame only if the road component
   touching the vehicle is confident and large enough.
3. The baseline planner takes the centre of each sampled row, fits one curve, and
   blends it with the previous path on the ground.
4. Corridor and ground-projection checks reject paths that cross mask holes or reach
   past the projection's uncertainty bound.
5. Stanley steers with curvature feedforward and latency compensation, and the speed
   law caps cruise at 5.5 mph under a 6 mph limit.

Fallbacks, from mildest to strongest:

| Trigger                                          | Response                                            | Setting                                             |
| ------------------------------------------------ | --------------------------------------------------- | --------------------------------------------------- |
| Gate or path check fails                         | Hold the last good path at 0.4x speed for 10 frames | `planning.gate.hold_frames`, `hold_speed_scale`     |
| Still failing after the hold                     | Stop: empty path, zero speed, brake                 | `planning.gate`                                     |
| Frame older than 200 ms or from the future       | Reset the planner and stop for that frame           | `runtime.max_frame_age_ms`                          |
| Under 1.5% road for 2 s, stuck 3 s, stale motion | Safe stop: park, manual control until `P` resumes   | `safety.min_road_fraction`, `safety.no_road_time_s` |

## Experiments

The bird's-eye grid and advanced planners, MPC and MPPI steering, the RL residual and
the imitation learning pipeline are in [Experiments](experiment/README.md). They reuse
this stack's loop and safe stop through `python -m experiment.run`, keep their settings
in `experiment/configs/`, and are not part of the installed package. The default
loader refuses an experiment config, so a run cannot silently fall back to the baseline.

## Build

```bash
python -m build
```

Creates the wheel and source archive in `dist/`.

## Testing

| Command                                   | What It Runs                                                | Needs BeamNG |
| ----------------------------------------- | ----------------------------------------------------------- | ------------ |
| `pytest`                                  | Unit and integration tests with mocked models and simulator | No           |
| `pytest --cov`                            | The same suite with a 70% coverage gate                     | No           |
| `python scripts/sbend_sim.py`             | Existing controller regression on a synthetic S-bend        | No           |
| `pytest experiment/tests`                 | Experiment tests; not part of the coverage gate             | No           |
| `offroad-autonomy --benchmark-seconds 30` | Live timing benchmark                                       | Yes          |

## Common Commands

| Command                                 | Purpose                                            |
| --------------------------------------- | -------------------------------------------------- |
| `ruff check src tests scripts`          | Check lint                                         |
| `ruff format --check src tests scripts` | Check formatting                                   |
| `offroad-autonomy --record-video`       | Record the dashboard; requires ffmpeg with libx264 |
| `offroad-autonomy --presentation-view`  | Show the presentation layout                       |
| `python scripts/derive_ego_mask.py`     | Regenerate hood exclusion after camera changes     |

## Development Notes

- Package code is in `src/offroad_autonomy/`; `pipeline.py` orders the stages.
- BeamNG dependencies stay in `simulation/`. RGB-only callers use `RGBPlanningPipeline`.
- Tuning belongs in `configs/default.yaml`; platform-specific simulator settings resolve automatically.
- Optional visible-path preview and cropped-road fit settings live under `control` and `planning`.
- CUDA FP16 is configurable with `perception.half`. Supported Windows GUIs render asynchronously.
- Camera mount or lens changes require recalibration and a new ego mask.

Current validation does **not** meet the 0.85 mIoU, sustained 20 FPS, or 50 ms
latency targets. Ingestion of 28 unique camera exposures per second is not certified.

## Contribution Rules

- Create a branch from `main` and open a focused pull request; do not commit directly to `main`.
- Run lint, formatting, tests with coverage, the S-bend regression, and the build before review.
- Do not create commits unless explicitly asked.
