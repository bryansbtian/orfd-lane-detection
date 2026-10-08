# Off-Road Autonomy

## Introduction

Off-Road Autonomy drives a vehicle along unmarked dirt trails in BeamNG.tech using one dashcam. It segments the drivable trail with a YOLOE-26 model, plans a centreline and steers with Stanley control.

Perception uses RGB segmentation only. There is no depth estimation, terrain fusion or depth-based obstacle veto. One GMSL2 dashcam with 120° horizontal FOV supplies segmentation, planning, control and the dashboard. It retains the previous dashcam mount: height 1.85 m, forward offset -0.30 m, pitch -8°. Capture is 960x620 at 30 FPS, processed at 720x465. The dashboard draws the same captured frame and its results; it does not poll another camera. Debug views are `0` path, `1` raw dashcam, `6` mask and `9` pipeline. Configure the sensor, mount and hood exclusion under `beamng.camera`. Ground distances still use a flat-ground assumption. Old `beamng.cameras`, `segmentation_mode` and `stitching` settings must be removed.

It runs on Windows, WSL2, macOS or a Jetson. BeamNG.tech itself runs only on Windows, so every other machine connects to it over the network.

When the hood hides the nearest part of the path, steering joins a visible lookahead point with a geometric pursuit arc. It does not extrapolate the path backward under the hood, which could reverse the commanded turn. Stanley tracking resumes when the predicted vehicle position lies within the observed path.

## Setup

Requires Python 3.10+ and BeamNG.tech 0.38 (`beamngpy` 1.35).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Put the model weights in `models/`: `yoloe-26x-seg.pt` (default config), `yoloe-26s-seg.pt` (Jetson) and `mobileclip2_b.ts` (the YOLOE text encoder).

### Custom Semantic Weights

Custom semantic segmentation checkpoints such as `models/weights.pt` require
Ultralytics 8.4.55 or newer (`pip install -e ".[dev]"`). Set `perception.model_weights`
to the checkpoint and include its road class name in `perception.prompts`.
For semantic models these names select fixed classes; other classes are excluded.
The confidence readout is the mean predicted probability over accepted road pixels
outside the ego mask. YOLOE instance segmentation weights remain supported.

### Connect To BeamNG

BeamNG.tech runs only on Windows. The Windows machine running it is the BeamNG host. The stack runs either on the BeamNG host or on another machine that connects to it.

| Stack Runs On                | What To Set                                                  | How To Start BeamNG      |
| ---------------------------- | ------------------------------------------------------------ | ------------------------ |
| Windows, on the BeamNG host  | `BEAMNG_HOME` to launch automatically, or start the server below | BeamNGpy server enabled |
| WSL2, on the BeamNG host     | Nothing                                                      | Listening on the network |
| macOS                        | `BEAMNG_HOST` (the BeamNG host's address)                    | Listening on the network |
| Jetson                       | `BEAMNG_HOST` (the BeamNG host's address)                    | Listening on the network |

For automatic launch on Windows, set the install folder in the same PowerShell session:

```powershell
$env:BEAMNG_HOME = "<your BeamNG.tech folder>"
$env:BEAMNG_LAUNCH = "true"
offroad-autonomy
```

The folder must contain `Bin64\BeamNG.tech.x64.exe`. To retain the path for future
terminals, use `[Environment]::SetEnvironmentVariable("BEAMNG_HOME", $env:BEAMNG_HOME, "User")`.
For local Windows connections with `launch: auto`, the app also reads that saved
user setting when neither the current environment nor YAML supplies an install
folder. Existing terminals can therefore use it without restarting VS Code.

With no install folder configured, the app only attaches to an existing server.
Opening BeamNG normally does not enable that server. Start it manually with:

```powershell
cd "<your BeamNG.tech folder>"
.\Bin64\BeamNG.tech.x64.exe -nosteam -tcom -tport 64256 -tcom-listen-ip "127.0.0.1"
```

For WSL2, macOS and Jetson, BeamNG must listen on the network. Start it from PowerShell on the BeamNG host:

```powershell
cd "<your BeamNG.tech folder>"
.\Bin64\BeamNG.tech.x64.exe -nosteam -tcom -tport 64256 -tcom-listen-ip "*"
```

If the connection is refused, allow TCP port 64256 through the Windows firewall. For macOS and Jetson, find the BeamNG host's address with `ipconfig`.

The first log line shows the address the stack connects to. `BEAMNG_HOST` overrides it on any machine.
The app uses port **64256**; BeamNG's `-tcom` default is **25252**, so include
`-tport 64256` or set `BEAMNG_PORT` to the server's actual port. See the
[BeamNG connection arguments](https://documentation.beamng.com/beamng_tech/arguments_and_settings/).

## Run

```bash
offroad-autonomy
```

The pipeline uses Stanley steering with curvature feedforward and speed planning. Tune it
under `control` in `configs/default.yaml`. Remove the former `control.controller` key
and `--controller` flag from custom configurations and launch commands.

For the bird's-eye grid planner, run `offroad-autonomy --config configs/grid.yaml`.
At a split, the grid planner chooses the traversable branch closest to the current
trajectory and locks onto it while multiple branches remain. A change in branch scores
does not switch branches; the lock releases when only one option remains or the chosen
branch is no longer safe. Arcs within a branch still use the existing quality scores and
motion-compensated RMS trajectory-distance penalty. `planning.grid.weight_trajectory_distance`
sets that penalty per metre (default `0.35`, `0` disables the soft penalty, not branch locking).

### Grid Planner

`planning.mode: grid` projects the road mask onto flat ground, fuses it over time using the
vehicle's own motion, and picks the best of a fan of steerable arcs on that grid
(`planning/bev_grid.py`, `planning/arc_planner.py`). It was built for the roof-mounted dashcam,
whose hood hides the first ~3 m of ground and whose image edges cut wide trails:

- Ground the camera cannot see is unknown, never "not road", so a trail running off the side of
  the image does not pull the path toward the visible part.
- Memory carries the road the hood now hides, using the simulator's direction vector to follow
  the vehicle's motion. Without a valid pose the grid forgets rather than smearing.
- Frames the perception gate rejects move the grid with the vehicle but add no evidence.
- Arcs end where the vehicle's body would leave the road, and the path only extends as far as
  the grid has seen road, so the path-end speed law still applies.
- The projection assumes flat ground: on crests and in dips far cells are misplaced, and the
  grid has no height information, so obstacles the segmenter labels as road are not seen.

Settings, with reasons, are in `planning.grid` in `configs/default.yaml`. The baseline planner
stays the default until the grid planner has been compared on several maps.

The grid's road evidence comes only from the learned dashcam mask. BeamNG annotation,
instance-label and depth buffers are explicitly disabled; the optional orbit RGB camera
is display-only. `semantic_mask` in the model wrapper means a neural-network prediction,
not a simulator annotation. The grid does use BeamNG's true vehicle position and direction
to register its memory, and Stanley uses simulator velocity/speed. This is vision-based
traversability with simulator-provided localization, so it does not measure performance
with noisy estimated odometry. The synthetic S-bend and planner tests use idealized masks
to test mechanics and are not evidence of real perception accuracy.

False positives remain possible: a mistaken road mask can accumulate in grid memory,
and flat-ground projection can misplace terrain on slopes. Unknown cells are not labeled
as observed road, but the clearance calculation treats them optimistically so unseen
ground under the hood does not block all motion. Neither that clearance nor a large
positive road score guarantees that the physical terrain is safe. Ground-truth road
labels would need to remain in a separate evaluator when measuring perception accuracy.

The BeamNG client selects realistic shifting and forward drive on startup and
resume, so holding the brake cannot engage arcade reverse. Manual gearboxes use
first gear for low-speed off-road operation; automatics select Drive. Gate holds
show their reason in the dashboard and console. The default gate accepts a
connected road at confidence 0.18 or above and still rejects insufficient or
disconnected road masks. Run `python scripts/check_brake_hold.py --reproduce-arcade`
to reproduce the former reversal and verify the brake-hold/forward-restart fix.

- Starts BeamNG.tech when `BEAMNG_HOME` is set and it is not already running, then spawns the vehicle and opens the dashboard
- `E` safe stop, `P` resume, `0`, `1`, `6`, `9` debug views, `T` timing overlay, `Q` quit
- `--headless` runs without a window
- `--presentation-view` shows the orbit camera, dashcam overlay and stats together live
- `--record-video` saves the dashboard to an mp4 (see Record The Dashboard below)
- `--presentation` saves a 1920 x 1080 video for an audience outside the team (see Record A Presentation below)

### Live Presentation View

```powershell
offroad-autonomy --presentation-view
```

Opens the split view with the orbit camera on the left and dashcam overlay and
telemetry on the right. No video is recorded and ffmpeg is not required. Use
`E` to stop, `P` to resume and `Q` to quit. Add `--presentation` to record this
view too. `--headless` suppresses the live window.

### Record The Dashboard

```bash
offroad-autonomy --record-video
offroad-autonomy --record-video --headless --benchmark-seconds 60 --label stanley-run
```

`--record-video` saves the dashboard, exactly as drawn, to `output/videos/<label>_<timestamp>.mp4`
(or the path given with `--record-video-out`). The file is H.264 in yuv420p with the index at the
front, so it plays in the VS Code media preview and in browsers.

- Needs `ffmpeg` with libx264 on `PATH` (`sudo apt install ffmpeg`). Without it the run stops before
  connecting to BeamNG. A snap-packaged ffmpeg cannot write under `/tmp`, so keep the output in
  your home directory.
- Encoding runs in a separate ffmpeg process fed from its own thread, so neither the control loop
  nor the dashboard waits for it. If the encoder falls behind, frames are dropped from the video
  and counted in the log when the run ends.
- Frames are paced onto a constant `recording.fps`, so the video plays back at real speed even
  when the dashboard stalls.
- With `--headless` the dashboard is drawn off screen for the recording only.
- Quality, preset and queue length are in the `recording` section of `configs/default.yaml`.

### Record A Presentation

```bash
offroad-autonomy --presentation
offroad-autonomy --presentation --headless --presentation-out ~/videos/demo.mp4
offroad-autonomy --presentation --record-video
```

`--presentation` records a 1920 x 1080 "Split Story" video to
`output/presentations/<label>_<timestamp>.mp4` (or the path given with `--presentation-out`), apart
from the dashboard recordings in `output/videos/`. It uses the same ffmpeg recorder and `recording`
settings as `--record-video`. Add `--record-video` to record the dashboard as well, to its own
file. The two paths must differ. The window, if there is one, still shows the operator dashboard,
unchanged.

- **Left:** a chase camera behind and above the vehicle, labelled Orbit Camera. Until its first
  frame arrives the view reads No Orbit Signal.
- **Top right:** the dashcam image with the traversable mask, planned path and ego exclusion, drawn
  by the same code as the dashboard's view.
- **Bottom right:** speed, steering, Autonomy FPS, latency mean / p95, Road / Valid Px and
  segmentation confidence, coloured by the dashboard's rules and thresholds.
- **Safe stop:** the header chip turns solid red, the orbit view gets a red wash and border, and no
  path is drawn on the dashcam image.

The orbit camera is attached when recording or showing a presentation, and no pipeline stage reads it. Its pose,
resolution and field of view are in the `presentation` section of `configs/default.yaml`. The main
loop polls it right after each dashcam frame, because the socket transport cannot be read from the
dashboard thread. That poll is logged as the `orbit_capture` stage and is part of the measured
primary loop, so the Autonomy FPS in the video is the real rate of the loop with the camera
attached. It has not yet been measured against BeamNG, so expect a lower FPS than without the flag
until it has.

Recording-only presentation frames are drawn on their own thread. The live presentation
uses the configured display threading mode; Tkinter requires drawing on the main thread.
Its render time is in the periodic runtime log line.

### Dashboard

The 1600 x 900 dashboard has a header with the autonomy status, the dashcam view with the traversable mask, planned path and ego exclusion, a Vehicle card, a Perception card and a Runtime strip. Speed is the largest number on screen.

- **Safe stop:** a red wash, a red border and a banner cover the dashcam view, and no planned path is drawn.
- **Autonomy FPS:** green at `visualization.dashboard.target_fps` or above, amber from `fps_warn_fraction` of it up to the target, red below that. Latency is red when its p95 passes `latency_budget_ms`.
- **Segmentation confidence:** red below the gate's `planning.gate.confidence_threshold`, amber up to `confidence_good`, green above.
- **Ticks:** the Road / Valid Px bar is marked at `safety.min_road_fraction` and the confidence bar at the gate threshold, both read from their own config sections.
- **Dashboard FPS:** shown in its own tile and never colored, because a slow window says nothing about the vehicle.
- **Held path:** when the planner holds a fallback path, it is drawn dashed in amber.

Text is drawn with Pillow using the bundled IBM Plex Sans and Mono fonts in `src/offroad_autonomy/visualization/fonts/` (SIL Open Font License, included). A missing font file stops startup.

## Run On Jetson (Docker)

Requires JetPack 6 with the NVIDIA container runtime. Start BeamNG listening on the network first (see [Connect To BeamNG](#connect-to-beamng)).

1. Build the image and export the TensorRT engine (first time only, since engines are tied to the device that builds them):

```bash
docker compose build
docker compose run --rm --entrypoint python autonomy scripts/export_engine.py --weights models/yoloe-26s-seg.pt
```

2. Run:

```bash
BEAMNG_HOST=192.168.1.50 docker compose up
```

- The container runs `configs/jetson.yaml`: TensorRT engine, headless
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
| `offroad-autonomy --record-video`                               | Run and save the dashboard to `output/videos/` as mp4   |
| `offroad-autonomy --presentation`                               | Run and save a presentation to `output/presentations/`  |
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
- `beamng.host`, `beamng.launch`, `beamng.camera_transport` and `ui.display_async` default to `auto` and are derived from the platform; `BEAMNG_HOST`, `BEAMNG_PORT`, `BEAMNG_HOME` and `BEAMNG_LAUNCH` override the `beamng:` block
- The dashboard runs on its own thread, and the control loop never waits for it
- After changing the mount or lens in `beamng.camera`, re-run `scripts/derive_ego_mask.py` and re-check `planning.roi_height`
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
