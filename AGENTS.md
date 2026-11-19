# Off-Road Autonomy

Autonomous off-road driving in BeamNG.tech: a YOLOE-26 model segments the drivable trail,
a planner fits a centreline on the ground, and a Stanley controller steers. The single
idea behind the design is that the control loop never waits for anything slow: the
dashboard runs on its own thread. Perception uses RGB only.

## Architecture

Keep these boundaries clear:

- `simulation/`: all BeamNG I/O. The only package that imports `beamngpy`.
- `perception/`: segmentation, camera geometry, ego masking.
- `planning/`: the perception gate, then the baseline planner and its hold/stop fallback.
- `control/`: Stanley steering and the speed law, all computed in metres on the ground.
- `runtime/`: the dashboard worker thread, timing, benchmarks.
- `visualization/`: the operator dashboard and window.
- `pipeline.py` orders the stages; `main.py` owns the loop, safe stop and shutdown.
- `experiment/` (outside the package): grid and advanced planners, MPC, MPPI, RL and
  imitation learning. Experiments subclass the default pipeline, planner and BeamNG
  client and run through `python -m experiment.run`; the package never imports them.

Rules that carry weight:

- Nothing outside `simulation/` may import `beamngpy`, so the stack stays testable without
  a simulator. The one exception is an experiment's `BeamNGClient` subclass, such as
  `experiment/imitation/evaluation_client.py`, which imports it lazily as the client does.
- The single GMSL2 dashcam supplies every stage. The dashboard consumes the same
  inference snapshot and never captures a separate image.
- `perception/__init__.py` must never import `RoadSegmenter`, because that pulls in Torch
  and Ultralytics for code that only needs geometry.
- The control loop must never block on the dashboard.
- Every tuning value lives in `configs/default.yaml` with its reason next to it, not as a
  literal in code. Experiment-only values live in `experiment/configs/` instead.
- The default run is the baseline planner with Stanley. A new planner or controller starts
  in `experiment/` and moves into the package only when it becomes the default.
- Machine-specific simulator settings stay `auto` and resolve in `utils/environment.py`;
  never write a host, install path or transport for one machine into a config.

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
