# Off-Road Autonomy

Autonomous off-road driving in BeamNG.tech: a YOLOE-26 model segments the drivable trail,
stereo depth vetoes obstacles, a planner fits a centreline on the ground, and a Stanley
controller steers. The single idea behind the design is that the control loop never waits
for anything slow: stereo and the dashboard run on their own threads, and missing depth
costs accuracy, never control.

## Architecture

Keep these boundaries clear:

- `simulation/`: all BeamNG I/O. The only package that imports `beamngpy`.
- `perception/`: segmentation, stereo rectification and depth, terrain, fusion.
- `planning/`: the perception gate, then the baseline or advanced planner.
- `control/`: Stanley steering and the speed law, all computed in metres on the ground.
- `runtime/`: the stereo and dashboard worker threads, timing, benchmarks.
- `visualization/`: the operator dashboard and window.
- `pipeline.py` orders the stages; `main.py` owns the loop, safe stop and shutdown.

Rules that carry weight:

- Nothing outside `simulation/` may import `beamngpy`, so the stack stays testable without
  a simulator.
- The display camera must never reach `AutonomyPipeline`, `PerceptionView` or
  `build_camera_models`; it is drawn on, never computed from.
- `perception/__init__.py` must never import `RoadSegmenter`, because that pulls in Torch
  and Ultralytics for code that only needs geometry.
- The control loop must never block on the stereo worker or the dashboard.
- Only triangulated pixels may raise an obstacle; ground-plane-inferred pixels never can.
- Every tuning value lives in `configs/default.yaml` with its reason next to it, not as a
  literal in code.

Prefer simple, well-defined boundaries over unnecessary abstractions or infrastructure.

## Non-Negotiable Style Rules

- Never use em dashes.
- Always use Title Case for user-facing titles and labels, such as README headings and
  workflow step names.
- Never use ternary expressions.
- Always use block-form `if` statements, even for single statements.

```python
# Correct
if ready:
    return

# Incorrect
if ready: return

# Incorrect
status = "Ready" if ready else "Pending"

# Correct
status = "Pending"
if ready:
    status = "Ready"
```

- Comments must explain **why**, not **what**.

```python
# Correct: BeamNG keeps applying the last command, so a client that vanishes leaves the vehicle driving.
self.park()

# Incorrect: Park the vehicle.
self.park()
```

## Engineering Principles

- Prefer readable control flow over clever or compact code.
- Keep simulator-specific logic isolated in `simulation/`.
- Keep secrets, credentials, tokens, and sensitive data out of source code, logs, stored
  files, and tests.
- Do not add infrastructure, dependencies, interfaces, or abstractions without a concrete
  need.
- Add tests for meaningful behavior and failure cases.
- Re-run `scripts/sbend_sim.py` after any change to the controller or its config.
- Keep changes scoped to the task being implemented.
- Update documentation when behavior or architecture changes.
- Do not claim unfinished functionality is implemented, and do not quote performance
  numbers that were not measured.

## Before Finishing

Run the relevant repository checks and fix all failures:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest --cov
python scripts/sbend_sim.py
python -m build
```

Review the final diff for unnecessary code, unused dependencies, style violations, secrets,
and accidental scope expansion.
