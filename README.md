# Off-Road Autonomy

## Introduction

Off-Road Autonomy drives a vehicle along unmarked dirt trails in BeamNG.tech using two front cameras. It segments the drivable trail with a YOLOE-26 model, measures obstacles with stereo depth, plans a centreline and steers with a Stanley controller.

It runs on a Windows workstation next to the simulator, or in a Docker container on an NVIDIA Jetson that connects to the simulator over the network.

## Development Setup

Requires Python 3.10+ and BeamNG.tech 0.38 (`beamngpy` 1.35).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Put the model weights in `models/`: `yoloe-26x-seg.pt` (default config), `yoloe-26s-seg.pt` (Jetson) and `mobileclip2_b.ts` (the YOLOE text encoder).

Set `beamng.home` in `configs/default.yaml` to your BeamNG.tech folder, or export `BEAMNG_HOME`.

## Run Locally

```bash
offroad-autonomy --config configs/default.yaml
```

- Launches BeamNG.tech if it is not already running, then spawns the vehicle and opens the dashboard
- `E` safe stop, `P` resume, `0`-`9` debug views, `T` timing overlay, `Q` quit
- `--headless` runs without a window

## Run On Jetson (Docker)

Requires JetPack 6 with the NVIDIA container runtime.

1. On the BeamNG machine, start the simulator listening on the network, and allow TCP port 64256 through the firewall:

```powershell
Bin64\BeamNG.tech.x64.exe -nosteam -tcom -tport 64256 -tcom-listen-ip "*"
```

2. On the Jetson, build the image and export the TensorRT engine (first time only, since engines are tied to the device that builds them):

```bash
docker compose build
docker compose run --rm --entrypoint python autonomy scripts/export_engine.py --weights models/yoloe-26s-seg.pt
```

3. Run:

```bash
BEAMNG_HOST=192.168.1.50 docker compose up
```

- The container runs `configs/jetson.yaml`: attach to the running simulator, socket camera transport, TensorRT engine, headless
- `models/`, `output/` and `configs/` are mounted from the host, so config changes need no rebuild
- `docker compose stop` parks the vehicle before disconnecting

## Testing

| Command                                       | What it runs                                                 | Needs BeamNG |
| --------------------------------------------- | ------------------------------------------------------------ | ------------ |
| `pytest`                                      | Unit and integration tests (models and simulator are mocked) | No           |
| `pytest --cov`                                | Same suite plus the 70% coverage gate CI enforces            | No           |
| `docker compose --profile test run --rm test` | The same suite inside the Jetson image                       | No           |
| `python scripts/sbend_sim.py`                 | Closed-loop controller check on a synthetic S-bend           | No           |
| `offroad-autonomy --benchmark-seconds 90`     | Timed run that writes `output/benchmarks/<label>.json`       | Yes          |

## Common Commands

| Command                                                         | What it does                                            |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| `offroad-autonomy --config <file>`                              | Run the stack with a config                             |
| `ruff check src tests scripts`                                  | Lint                                                    |
| `ruff format src tests scripts`                                 | Format (CI runs it with `--check`)                      |
| `python -m build`                                               | Build the wheel and sdist                               |
| `pytest`                                                        | Run the tests                                           |
| `python scripts/compare_benchmarks.py output/benchmarks/*.json` | Compare benchmark runs side by side                     |
| `python scripts/derive_ego_mask.py`                             | Regenerate camera bodywork masks after moving a camera  |
| `python scripts/diagnose_perception.py capture` / `analyze`     | Save simulator frames, then dump every perception stage |
| `python scripts/closed_loop_log.py --seconds 120 --summary`     | Headless run that logs every control frame to CSV       |
| `python scripts/export_engine.py`                               | Export the segmentation model to TensorRT               |

## Development Notes

- Package code is in `src/offroad_autonomy/`; the entry point is `main.py`, and the stage order is in `pipeline.py`
- Only `simulation/beamng_client.py` imports `beamngpy`
- `configs/default.yaml` holds every tuning value, with the reason for it next to the key; `configs/jetson.yaml` overrides it through `extends:`
- `BEAMNG_HOST`, `BEAMNG_PORT`, `BEAMNG_HOME` and `BEAMNG_LAUNCH` override the `beamng:` block
- Stereo and the dashboard run on their own threads, and the control loop never waits for either
- After moving a camera in `beamng.cameras`, re-run `scripts/derive_ego_mask.py` and re-check `planning.roi_height` and `depth.min_depth_m`
- Manual driving (W/A/S/D under safe stop) reads the OS key state and works only on Windows
- CI (`.github/workflows/ci.yml`) runs lint, format, the S-bend check, tests with coverage and the build on Python 3.10; CodeQL scans weekly and on every pull request; Dependabot watches pip, the Docker base image and the Actions

## Contribution Rules

- Create a new branch from main for every change.
- Do not commit directly to main.
- Open a pull request into main when the change is ready.
- Keep pull requests small, focused, and easy to review.
- Run `ruff check src tests scripts`, `ruff format --check src tests scripts` and `pytest --cov` before opening a pull request.
- Run `python scripts/sbend_sim.py` after changing the controller or its config.
- No ternary expressions, and no single-line `if` bodies.
- Comments explain why, not what.
- Do not create commits unless explicitly asked.
- Before finishing, summarize what changed, what commands were run, what commands could not be run, and any remaining risks.
