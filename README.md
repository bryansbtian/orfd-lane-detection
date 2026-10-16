# Off-Road Autonomy

## Introduction

RGB road segmentation and trajectory planning for unmarked trails, with BeamNG.tech
for simulation testing. YOLO26 predicts traversable road; the planner generates a
trajectory and validity status. The existing Stanley controller handles simulation actuation.

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

The simulator connection uses port **64256**. For an existing server or a remote
Windows host, see [Connection Setup](docs/usage.md#connect-to-beamng).

Dashboard keys: `E` safe stop, `P` resume, `0/1/6/9` debug views, `T` timing, `Q` quit.
Use `--headless` to run without a window or `--config <file>` for a config overlay.

## Build

```bash
python -m build
```

Creates the wheel and source archive in `dist/`.

## Testing

| Command | What It Runs | Needs BeamNG |
| --- | --- | --- |
| `pytest` | Unit and integration tests with mocked models and simulator | No |
| `pytest --cov` | The same suite with a 70% coverage gate | No |
| `python scripts/sbend_sim.py` | Existing controller regression on a synthetic S-bend | No |
| `offroad-autonomy --benchmark-seconds 30` | Live timing benchmark | Yes |

## Common Commands

| Command | Purpose |
| --- | --- |
| `ruff check src tests scripts` | Check lint |
| `ruff format --check src tests scripts` | Check formatting |
| `offroad-autonomy --record-video` | Record the dashboard; requires ffmpeg with libx264 |
| `offroad-autonomy --presentation-view` | Show the presentation layout |
| `python scripts/derive_ego_mask.py` | Regenerate hood exclusion after camera changes |

## Development Notes

- Package code is in `src/offroad_autonomy/`; `pipeline.py` orders the stages.
- BeamNG dependencies stay in `simulation/`. RGB-only callers use `RGBPlanningPipeline`.
- Tuning belongs in `configs/default.yaml`; platform-specific simulator settings resolve automatically.
- CUDA FP16 is configurable with `perception.half`. Supported Windows GUIs render asynchronously.
- Camera mount or lens changes require recalibration and a new ego mask.
- [Usage And Architecture](docs/usage.md) covers remote connections, recording, and Jetson setup.
- [RGB Pipeline Audit](docs/rgb_pipeline_audit.md) covers accuracy, planning validation, and sim-to-real limits.
- [Performance Results](docs/fps_optimization.md) covers measured improvements and reproducible benchmarks.

Current validation does **not** meet the 0.85 mIoU, sustained 20 FPS, or 50 ms
latency targets. Ingestion of 28 unique camera exposures per second is not certified.

## Contribution Rules

- Create a branch from `main` and open a focused pull request; do not commit directly to `main`.
- Follow [CLAUDE.md](CLAUDE.md): no ternaries, inline `if` bodies, or em dashes; comments explain why.
- Run lint, formatting, tests with coverage, the S-bend regression, and the build before review.
- Do not create commits unless explicitly asked.
