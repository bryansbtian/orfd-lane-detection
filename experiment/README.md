# Experiments

Alternatives to the default baseline planner and Stanley controller. None of this is
installed with the package or used by `offroad-autonomy`. Each experiment plugs into
the default stack by subclassing it, so it drives under the same loop, perception gate
fallback and safe stop. Run everything from the repository root.

| Folder        | What It Is                                                         | Config                          |
| ------------- | ------------------------------------------------------------------ | ------------------------------- |
| `grid/`       | Bird's-eye traversability grid with memory and an arc planner      | `configs/grid.yaml`             |
| `advanced/`   | Scored row segments, temporal blending and a Kalman prior          | `configs/advanced.yaml`         |
| `predictive/` | Delay-aware MPC and smooth MPPI steering on the grid planner       | `configs/grid_mpc.yaml`, `configs/grid_mppi.yaml` |
| `rl/`         | Residual steering correction learned on the S-bend model           | `configs/rl.yaml`               |
| `imitation/`  | Steering proxy learned from MCAP recordings                        | `configs/imitation.yaml`        |

## Run An Experiment

```powershell
python -m experiment.run --config experiment/configs/grid.yaml
python -m experiment.run --config experiment/configs/grid_mppi.yaml --record-video
python -m experiment.run --planner grid --controller mpc
```

Every `offroad-autonomy` flag works here. Each config extends
[Default Config](../configs/default.yaml) and adds an `experiment:` section that picks
`planner` (`baseline`, `advanced`, `grid`) and `controller` (`stanley`, `mpc`, `mppi`,
`rl`, `imitation`), plus the settings that experiment needs. `offroad-autonomy` refuses
these files rather than silently driving the baseline. Presentation videos are labelled
with the experiment being run.

## Test

```powershell
pytest experiment/tests
```

These tests are not in the default `pytest` run, its coverage gate or CI, so run them
after changing the default stack's planner, controller, pipeline or BeamNG client.

## Reinforcement Learning

Train an experimental steering policy offline, then select it for a BeamNG run:

```powershell
python -m experiment.rl.train
python -m experiment.run --config experiment/configs/rl.yaml
```

Training writes `models/rl_policy.npz` and a paired comparison against Stanley to
`output/rl/comparison.json`. It uses Augmented Random Search, learning from episode
rewards on the synthetic S-bend bicycle model in `scripts/sbend_sim.py`, without extra
dependencies. The reward penalizes tracking error, trail-edge exposure, steering changes
and failed completion. Bends in both directions, loop rates, actuation delays and initial
offsets vary. Policy selection uses training scenarios; the comparison uses separate seeds.

The policy observes RGB-derived path errors and curvature, speed, and previous steering.
It adds a bounded steering correction before Stanley's steering cap, smoothing and speed
law. Perception fallback and gate stops bypass the learned correction. A missing or
incompatible checkpoint fails before the simulator connection.

```powershell
python -m experiment.rl.train --iterations 30 --out models/rl_long.npz --report output/rl/long.json
python -m experiment.rl.train --evaluate models/rl_long.npz --report output/rl/recheck.json
python -m experiment.run --rl-policy models/rl_long.npz
```

Tune `control.rl` in `configs/rl.yaml`, keeping normalization and residual limits
consistent with the checkpoint. The offline model does not reproduce terrain, traction,
RGB segmentation failures or BeamNG dynamics, so an offline improvement does not
establish better driving in BeamNG. Compare matched routes and initial conditions there.

## Imitation Learning

Install the extra dependencies, then extract, train and evaluate:

```powershell
python -m pip install -e ".[dev,imitation]"
python -m experiment.imitation extract
python -m experiment.imitation train
python -m experiment.imitation evaluate
```

`extract` discovers root `bag*.zip` archives, streams their original MCAP members and
writes the dataset to `imitation.dataset_dir` (`datasets/imitation`). Archives containing
only derived exports are skipped. Recordings are split chronologically into training,
validation and test partitions with guard intervals and duplicate-frame checks. The
dataset directory must be empty; pass `--dataset` to make a new version.

```powershell
python -m experiment.imitation extract --archives recording_a.zip recording_b.zip --dataset datasets/imitation_v2
python -m experiment.imitation train --dataset datasets/imitation_v2 --device cuda
python -m experiment.imitation evaluate --dataset datasets/imitation_v2 --device cpu
```

The MobileNetV3 model consumes a front image and causal speed, and predicts a
measured-steering proxy. Radians and positive-yaw direction are supported by dynamics
checks, but publisher conversion and human control remain unverified. It does not learn
throttle or braking. The checkpoint is `models/imitation_best.pt`; curves, TensorBoard
events and prediction CSVs go to `output/imitation/baseline`.

Imitation actuation is simulation-only and requires a measured steering calibration.
Measure the parked Hopper's wheel response, then drive with it:

```powershell
python -m experiment.imitation calibrate
python -m experiment.run --config experiment/configs/imitation.yaml --imitation-experiment --imitation-calibration output/imitation/live_calibration/measured_calibration.json
python -m experiment.imitation compare
```

The version 2 adapter requires `--imitation-experiment`, binds to the configured vehicle
and camera, and caps speed at 2 m/s. It keeps the recording's unverified angle
conventions, assumed real wheelbase and missing real camera extrinsics. `compare` runs
Stanley, RL and imitation on the procedural roads in `configs/imitation_comparison.yaml`.

The last measured baseline beat zero steering on the held-out recording but lost to
previous-steering persistence, and its test route overlapped the training route.
