#!/usr/bin/env python3
"""Build The R3 Baseline Comparison From Preserved Measured Batches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offroad_autonomy.runtime.r3_evaluation import write_csv


def summaries(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("run_??.summary.json"))]


def yes_no(value):
    if value:
        return "Yes"
    return "No"


def verdict(value):
    if value:
        return "Pass"
    return "Fail"


def engineering_report(baseline, improved, output, comparison):
    base = json.loads((baseline / "aggregate.json").read_text())
    final = json.loads((improved / "aggregate.json").read_text())
    experiment = json.loads((improved / "experiment.json").read_text())
    selection = json.loads((ROOT / "results/r3/tracking_improvement/selection.json").read_text())
    audit = json.loads((improved / "data_audit.json").read_text())
    validation = json.loads((ROOT / "results/r3/tracking_improvement/validation.json").read_text())
    command_metrics = json.loads((output / "command_variation.json").read_text())
    command_variation = {
        group: float(
            np.mean(
                [row["total_variation_per_s"] for row in command_metrics if row["batch"] == group]
            )
        )
        for group in ("Original Baseline", "Improved Verification")
    }
    signed_means = {
        group: sum(
            row["mean_signed_offset_m"] * row["samples"]
            for row in command_metrics
            if row["batch"] == group
        )
        / sum(row["samples"] for row in command_metrics if row["batch"] == group)
        for group in ("Original Baseline", "Improved Verification")
    }
    first_run = json.loads((improved / "run_01.json").read_text())
    original_audit = json.loads((baseline / "data_audit.json").read_text())
    old_finish = [row["finish_distance_m"] for row in original_audit["runs"]]
    new_finish = [row["finish_distance_m"] for row in audit["runs"]]
    assert audit["all_five_integrity_checks_passed"] and audit["frozen_sources_match"]
    assert experiment["source_hashes"] == selection["source_hashes"]
    assert (
        experiment["experiment_phase"] == "verification" and not experiment["control_diagnostics"]
    )
    for path in improved.glob("run_??.json"):
        run = json.loads(path.read_text())
        assert run["source_hashes"] == selection["source_hashes"]
        assert datetime.fromisoformat(run["started_utc"]) > datetime.fromisoformat(
            selection["selected_utc"]
        )
    measurements = summaries(improved)
    route = experiment["route"]
    ref = "../" + improved.relative_to(ROOT).as_posix()
    figures = "../" + output.relative_to(ROOT).as_posix()
    passed = sum(s["all_proposed_test_conditions"] for s in measurements)
    mae_passes = sum(s["proposed_mean_target"] for s in measurements)
    sd_passes = sum(s["proposed_signed_sd_target"] for s in measurements)
    status = "Yellow"
    if any(
        not s["route_completed"]
        or s["road_departures"]
        or s["manual_interventions"]
        or not s["speed_compliance"]
        or not s["physics_recording_integrity"]
        for s in measurements
    ):
        status = "Red"
    rows = []
    development = []
    labels = {
        "01_baseline_diagnostic": "01: Baseline Replication",
        "02_no_blend": "02: No Temporal Blend",
        "03_preview6": "03: Minimum Preview 6 m",
        "04_visible_edges": "04: Visible Edges Only (Rejected)",
        "05_weighted_rows": "05: Cropped Row Weight 0.1 (Rejected)",
        "06_preview6_fast_steer": "06: Preview 6 m, Steering EMA 0.75",
        "07_preview6_curve03": "07: Preview 6 m, Curve Budget 0.3",
        "08_visible_origin": "08: Visible-Origin Adaptive Preview (Selected)",
        "09_visible_origin_tight_cte": "09: Visible-Origin, Deadzone 0.1 (Rejected)",
    }
    for row in comparison:
        if row["phase"] == "development":
            development.append(
                f"| {labels[row['batch']]} | {row['mean_absolute_lateral_offset_m']:.4f} | {row['std_signed_lateral_offset_m']:.4f} | {row['maximum_absolute_lateral_offset_m']:.4f} | {row['road_departures']} | {row['manual_interventions']} | {row['maximum_speed_mph']:.3f} | {yes_no(row['route_completed'])} |"
            )
    for s in measurements:
        rows.append(
            f"| {s['run']} | {s['mean_absolute_lateral_offset_m']:.4f} ({verdict(s['proposed_mean_target'])}) | {s['std_signed_lateral_offset_m']:.4f} ({verdict(s['proposed_signed_sd_target'])}) | {s['maximum_absolute_lateral_offset_m']:.4f} | {s['road_departures']} | {s['manual_interventions']} | {s['maximum_speed_mph']:.3f} | {yes_no(s['route_completed'])} | {s['duration_s']:.3f} | {verdict(s['all_proposed_test_conditions'])} |"
        )
    change_mean = 100 * (
        1
        - final["pooled_mean_absolute_lateral_offset_m"]
        / base["pooled_mean_absolute_lateral_offset_m"]
    )
    change_sd = 100 * (
        1 - final["pooled_std_signed_lateral_offset_m"] / base["pooled_std_signed_lateral_offset_m"]
    )
    text = f"""# R3: Closeness to Centerline (Simulation Only)

**Status: {status}.** Five original baseline trials and five independent trials of
the frozen candidate have been measured. The new batch completed
{final["completed_runs"]}/5 traversals; {passed}/5 met all proposed test conditions.
The 0.50 m centering limits remain proposed engineering targets awaiting mentor
approval. Physical trail-edge and reference-point definitions remain outstanding.
No sponsor-approved R3 compliance is claimed.

## Acceptance Criteria

| Metric | Criterion | Standing |
| --- | --- | --- |
| Mean Absolute Lateral Offset | At Most 0.50 m In Each Run | Proposed; Mentor Approval Pending |
| Population SD Of Signed Offset | At Most 0.50 m In Each Run | Proposed; Mentor Approval Pending |
| Road Departures | Zero | Confirmed; Physical Trail-Edge Definition Pending |
| Manual Interventions | Zero | Confirmed |
| Maximum Actual Speed | At Most 10 mph | Confirmed |
| Completion | Entire Approved A-To-B Course | Confirmed Open Traversal |
| Conditions | Hopper, Unmarked Dirt Road, Sunny Daytime, Fully Autonomous | Fixed |

Maximum offset is diagnostic without an invented acceptance threshold. A
completed simulation alone does not verify centering. Authored-road containment
and physical trail-edge approval are distinguished throughout this report.

## Current Implementation Status

The actual pipeline is one RGB camera, uniform preprocessing, fixed-class semantic
segmentation, optional mask postprocessing, a gated ground-plane centerline planner,
and the Stanley controller with its existing pursuit fallback, curve speed law,
and steering smoothing. Production motion comes from wheel/gyro odometry.
Optional MPC and MPPI implementations exist. Their predictive solve requires
`ground_path` and `ground_curvature`, produced by the grid planner. The baseline
image planner does not provide those fields, so switching only `control.mode`
would retain a Stanley fallback rather than provide a valid MPC/MPPI comparison.
No such switch is presented as an executed predictive-controller experiment.
No ROS 2 node or launch package is present in this checkout. External ROS
transport and executor behavior are outside this measured BeamNG path.

Selected configuration: `{selection["selected_config"]}`. The candidate was chosen
from development data and frozen before the five new attempts. The complete
[Selection Record](../results/r3/tracking_improvement/selection.json) records its
rationale, source/configuration hashes, and selection timestamp. No final trial
was used for subsequent tuning. The default driving configuration retains the
original active tuning; the candidate is a separate reproducible overlay.
No world pose, road station, or ground-truth geometry enters the control pipeline.

{selection["behavior_change"]}

The native 2,000 Hz footprint observer and source-whitelisted control authority
are reused. Qualification attempt 03's eight deliberate challenges detected and
blocked nine nonzero command events; those events are excluded from scored runs.
Qualification covers native keyboard/gamepad binding paths and ordinary driving
APIs, not physical gamepad hardware or adversarial process modification.

## Root-Cause Investigation

The original five-run pooled mean absolute error was 0.6519 m, signed SD 0.5587 m,
and maximum error 2.4835 m. [Peak Locations](../results/r3/tracking_improvement/baseline_analysis/peak_errors.csv)
place the excursions around stations 222.4 to 236.9 m on the complete road, near
the western-to-middle bend transition. Actual steering and speed at each peak
are preserved. The original batch did not record planner or confidence traces,
so those missing values were not retrospectively invented.

A separate baseline development run reproduced a 2.4356 m excursion. All 1,071
logged control frames used the existing pursuit fallback because the first
visible path point was 3.2189 m ahead, beyond the latency prediction position.
That branch overwrites the Stanley/feedforward steering formula. A synthetic
regression confirms that changing Stanley gain/feedforward alone does not change
its command. Curvature-adaptive lookahead and curve speed limits already existed.

At the worst diagnostic frame, segmentation confidence was 0.9751 while actual
reference-point offset was -2.4354 m. Confidence therefore did not establish
correct centerline geometry. Cropped road runs can span the viewport; their
midpoint is a field-of-view midpoint rather than an observed road center.
Synthetic cropped-road tests reproduce that bias. The [RGB And Mask Snapshots](../results/r3/tracking_improvement/development/01_baseline_diagnostic/run_01.control/)
and [Control Diagnostics](../results/r3/tracking_improvement/development/01_baseline_diagnostic/analysis/run_01.control.png)
support inspection. Projected-path world errors use approximate capture/pose
alignment and are diagnostic, not primary metrology. The first diagnostic run
used nominal mount coordinates; later runs use measured parked camera coordinates.

The measured camera mount is approximately 0.568 m forward, 0.551 m left, and
1.380 m up from BeamNG's native vehicle reference position. Its lateral offset
roughly matches the calibrated body-envelope center. Thus native reference
position and physical body center are different datums. The primary R3 metric
continues to use the original native reference position in every batch. No
coordinate shift or lateral target bias was introduced to make the scores pass.
Protocol agreement on the reference point remains necessary.
The original pooled signed mean is -0.5959 m; the body/camera displacement can
contribute to this persistent bias, but it does not explain all curved-road
excursions or remove their measured signed variation.

The baseline diagnostic pipeline averaged 37.3 ms; host-receive frame age averaged
49.5 ms. Median full control period was 0.1885 simulation seconds. These are
different from exposure-to-wheel response, which is not directly measured.
The 0.45 steering EMA adds command delay, but wheel angle and actuator calibration
remain unmeasured. Descriptive frame correlations with absolute error were 0.384
for absolute road curvature, 0.264 for absolute steering, -0.179 for speed, and
-0.388 for confidence. These autocorrelated associations do not prove causation.
Neither controller gain nor overspeed alone explains the observed errors.

## Testing Methodology

Map `automation_test_track`, Hopper `/vehicles/hopper/lxt6_M.pc`, road `23528.0`,
persistent ID `ccc97201-4b41-4e28-8f93-afd97e6c2d41`, material `m_dirt_road_01`.
Actual exported nodes {route["start_node_index"]} through {route["finish_node_index"]}
define the **{route["route_length_m"]:.12f} m** horizontal course.

| Parameter | Value |
| --- | --- |
| Simulator Version | `{first_run["beamng_version"]}` |
| Python | {validation["environment"]["python_version"]} In The Recorded Test Environment |
| A XYZ (m) | `{route["point_a_xyz_m"]}` |
| B XYZ (m) | `{route["point_b_xyz_m"]}` |
| Spawn XYZ (m) | `{route["spawn_position"]}` |
| Spawn Quaternion XYZW | `{route["spawn_rotation"]}` |
| Direction | West Around Western Bend, East Through Middle, Around Eastern Return Bend, Then West To B |
| Gates | A Within 1.50 m XY And Heading Cosine At Least 0.95; B Station Reached Within 2.00 m XY |
| Weather And Light | Sunny-Noon Preset; Frozen Daytime |
| Seeds | Python, NumPy, Torch And CUDA Seed 0; Simulator RNG Not Controlled |
| Warmup | Two Parked Inference Frames; Planning/Control And Local Odometry Reset Before Scoring |
| Packages | {"; ".join(f"{k} {v}" for k, v in experiment["versions"].items())} |

The [Actual Road Geometry](../results/r3/final_verification/approved_route_geometry.json)
preserves all 210 left/middle/right XYZ cross sections. Live geometry is checked
against this snapshot. No connecting road or predicted ground truth is invented.
The authored dirt ribbon is the test corridor; independent physical trail-edge
approval remains pending. Full details are in [The Methodology Appendix](R3_Verification_Methodology.md).

At every native physics step, shortest world-XY segment distance gives signed
offset, positive left along the route. Mean absolute error is `mean(abs(e))`;
signed SD is `std(e, ddof=0)`. Absolute-error SD is computed separately. Actual
XYZ velocity supplies speed. Complete convex footprint containment uses the
calibrated nominal body OOBB plus all 854 current JBeam nodes and a 0.05 m margin.
Departure episodes count contiguous outside intervals. Missing authority or
physics records cannot verify zero intervention. The score ends at the recorded
B gate; later raw observations remain preserved. This observes physics states,
without claiming an analytical continuous sweep between integration steps.

Every development attempt below was retained. Incomplete/stationary gate-stop
attempts are failures and their centering averages are not used to rank successful
candidates. The five new trials are separate from selection runs. Source hashes,
resolved settings, versions, actual initial conditions, commands, interventions,
footprints, and per-frame diagnostics are retained.

## Development Results

| Attempt | Mean Absolute (m) | Signed SD (m) | Maximum (m) | Departures | Manual Events | Maximum Speed (mph) | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(development)}

`01` is an instrumented baseline replication. `02` removes only the temporal
path blend. `03` raises only minimum preview to 6 m. `04` requires both visible
edges and is rejected when it cannot complete. Further configurations and the
selected candidate's rationale are enumerated in the selection record. Candidate
changes are general control/perception settings, without route-specific steering
tables, road-map feedback, or fixed lateral offsets.

Attempt 04 ceased progressing around 108 m because the perception gate had no
valid path. A later `cycleGearboxModes` input was detected and blocked, and the
runner stopped with one recorded intervention. Both failures are retained. Its
long stationary interval makes the apparent signed-SD target result unsuitable
as centering verification. The weighted-row attempt worsened both centering
metrics. Faster EMA and lower curve budget provided little additional benefit
over fixed preview; the smaller deadzone increased signed variability.

## Results

| Aggregate Metric | Original Baseline | Frozen Candidate |
| --- | ---: | ---: |
| Pooled Mean Absolute Offset (m) | {base["pooled_mean_absolute_lateral_offset_m"]:.4f} | {final["pooled_mean_absolute_lateral_offset_m"]:.4f} |
| Pooled Signed Mean (m) | {signed_means["Original Baseline"]:.4f} | {signed_means["Improved Verification"]:.4f} |
| Pooled Signed-Offset SD (m) | {base["pooled_std_signed_lateral_offset_m"]:.4f} | {final["pooled_std_signed_lateral_offset_m"]:.4f} |
| Pooled Absolute-Error SD (m) | {base["pooled_std_absolute_lateral_offset_m"]:.4f} | {final["pooled_std_absolute_lateral_offset_m"]:.4f} |
| Maximum Absolute Offset (m) | {base["maximum_absolute_lateral_offset_m"]:.4f} | {final["maximum_absolute_lateral_offset_m"]:.4f} |
| Departures From Authored Corridor | {base["road_departures"]} | {final["road_departures"]} |
| Recorded Interventions | {base["manual_interventions"]} | {final["manual_interventions"]} |
| Maximum Speed (mph) | {base["maximum_speed_mph"]:.3f} | {final["maximum_speed_mph"]:.3f} |
| Completed Routes | {base["completed_runs"]}/5 | {final["completed_runs"]}/5 |
| Runs Meeting All Proposed Conditions | {base["runs_meeting_all_proposed_conditions"]}/5 | {final["runs_meeting_all_proposed_conditions"]}/5 |
| Mean Command Variation Rate (Normalized Steering/s) | {command_variation["Original Baseline"]:.4f} | {command_variation["Improved Verification"]:.4f} |

| New Run | Mean Absolute (m) | Signed SD (m) | Maximum (m) | Departures | Manual Events | Maximum Speed (mph) | Completed | Duration (s) | All Proposed Conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
{chr(10).join(rows)}

Mean error met its proposed limit in {mae_passes}/5 new attempts; signed SD met its
proposed limit in {sd_passes}/5. Full-precision values and individual assessments
are in [The New Summary]({ref}/summary.csv). Pooled fixed-rate samples are duration
weighted and temporally correlated. Pooled SD includes between-run bias and is
not the average of individual SDs; the experimental repeats are five runs.

![Baseline Versus Independent Verification]({figures}/baseline_vs_improved.png)

Each point in the comparison is one complete run; black ticks are unweighted
means of the five per-run metrics. The aggregate table instead pools fixed-rate
physics samples. Neither is a confidence interval.

![Signed Error Along The Same Course]({figures}/baseline_vs_improved_offset.png)

![Signed Offset Over Time And Authored Boundaries]({ref}/lateral_offset_over_time.png)

![Mean Error And Signed SD With Provisional Targets]({ref}/lateral_offset_comparison.png)

![Numerical Performance Summary]({ref}/performance_summary.png)

![Approved Route And All New Measured Trajectories]({ref}/route_map.png)

Figure error bars are population SD of absolute error, distinct from signed SD
and confidence intervals. All plotted driving values are measured. Synthetic
curves are recorded separately in
[The Generalization Check](../results/r3/tracking_improvement/synthetic_generalization.json).
The visibility-aware alternative is compared with the baseline in
[The Adaptive Preview Check](../results/r3/tracking_improvement/adaptive_generalization.json).
That check spans 9, 15, and 25 m radii, 3 and 5 FPS, and two noise seeds. It tests
stability using idealized noisy paths, without real segmentation, uneven terrain,
or a physical footprint observer. Its limits and any regression relative to the
baseline must be reviewed; it is not additional BeamNG verification.

## Results Interpretation

Relative pooled mean-error reduction is {change_mean:.1f}%; signed-SD reduction
is {change_sd:.1f}%. Safety, completion, and provisional centering are separate
assessments. {passed}/5 new attempts met all proposed conditions. Final compliance
cannot follow from pooled averages when a per-run limit is missed, or while the
targets and physical boundary/reference definitions remain unapproved.

The experiments narrow the likely causes to path observability/estimation and
closed-loop response rather than an assumed lack of Stanley gain. High confidence
can coexist with displaced inferred geometry. A longer preview can reduce
near-field error and command variation, but may cut tighter ideal curves; this
tradeoff limits generalization claims. The reference-position offset also affects
interpretation of persistent signed bias and is not corrected after scoring.

The new pooled signed mean is {signed_means["Improved Verification"]:.4f} m,
close in magnitude to the measured camera/body displacement from the native
reference position. This supports a datum contribution to persistent bias; it
does not establish a body-center error metric. No constant shift is applied:
heading, roll, and curvature would require an explicit geometric transformation
and consistent re-evaluation of every batch under an approved datum definition.

The improvement is not uniform along the course. Recorded XY distance to B at
the finish gate increased from {min(old_finish):.3f} to {max(old_finish):.3f} m
across baseline runs to {min(new_finish):.3f} to {max(new_finish):.3f} m across
candidate runs. All remain within the existing 2.00 m completion gate. This
endpoint tradeoff is visible in the station comparison and must not be hidden
by improved whole-run averages. Completion does not establish precise endpoint
centering; the final gate and remaining end-of-course bias need review.

Command variation is `sum(abs(diff(command))) / elapsed_time`, calculated from
every scored physics sample, then averaged across runs. It describes command
activity, not measured wheel-angle slew or an approved oscillation threshold.
Full per-run values are in [The Command Comparison]({figures}/command_variation.csv).

## Challenges And Limitations

The approved course is one map, vehicle, direction, speed range, and warmed
initial condition. Independent rendering/inference timing is not bitwise
deterministic. The diagnostic recorder adds runtime work, so its baseline
replication is distinct from the original baseline. The final batch uses native
recording without the extra per-frame diagnostic query, matching the original
baseline's recording protocol. Planner and confidence traces belong to development
runs. Actual exposure timestamps and wheel
actuation are not available; apparent steering delay cannot be attributed fully
to the controller. Flat-ground projection, slope/bank, segmentation widths,
vehicle geometry estimates, and physical-road interpretation remain limitations.

The physical trail edge, vehicle reference point, and exact gates need protocol
confirmation. Authored-ribbon containment does not verify a differently defined
trail edge. Synthetic checks and millions of correlated samples do not replace
independent BeamNG trials on additional curved roads. MPC/MPPI would still depend
on the same perception geometry and are not presumed to fix its errors.

## Next Steps

1. Confirm or revise the proposed targets and sign off the boundary, gates, and
   vehicle datum. Preserve the existing datum for comparisons unless a new
   protocol explicitly requires re-evaluating all batches consistently.
2. Review remaining excursions with the saved RGB/mask/planner/control traces.
   Validate ground-relative camera geometry and actuator response independently.
3. Test additional curved dirt-road geometries and directions in BeamNG before
   promoting an overlay as a general replacement for the default configuration.
4. Keep development and verification separate. Any subsequent change requires a
   new freeze and a complete new verification batch, retaining failures.

## Reproducibility And Evidence

The original five-run batch and its report are preserved. The previous short
16.596 m preliminary runs remain separate and are excluded from these statistics.
[Archived Baseline Source And Report](../results/r3/tracking_improvement/baseline_source/)
and [All Attempt Metrics]({figures}/all_attempts.csv) retain unfavorable results.
[New Batch Manifest]({ref}/experiment.json) contains frozen provenance. Native
records, derived timestep CSVs, per-run summaries, and figures live together.
The native Lua observer must match the previously passed qualification hash.

The [Independent Data Audit]({ref}/data_audit.json) checks every recorded sequence,
time interval, command, authority counter, envelope and finish gate; it also checks
41 scalar geometry samples per run, including each maximum-error sample. Native
and scored CSV archives are lossless and hash verified. Plain CSVs remain locally;
transfer checkouts restore them from `.csv.gz` before regenerating plots.

[Software Validation](../results/r3/tracking_improvement/validation.json) records
{validation["pytest"]["passed"]} passing tests, {validation["pytest"]["skipped"]} skips,
{validation["pytest"]["coverage_percent"]:.2f}% coverage, and passing lint, formatting,
synthetic regression, and wheel/source-package builds. Software checks do not
replace live driving evidence.

```powershell
python -m pip install -e ".[dev,r3]"
python scripts/r3_verify.py run --runs 1 --phase development --diagnostics --snapshots --config {selection["selected_config"]} --out results/r3/tracking_improvement/development_new
python scripts/r3_verify.py run --runs 5 --phase verification --config {selection["selected_config"]} --out results/r3/tracking_improvement/verification_new
python scripts/r3_verify.py analyze --out results/r3/tracking_improvement/verification_new
python scripts/audit_r3_verification.py --batch results/r3/tracking_improvement/verification_new --compress-raw
python scripts/write_r3_tracking_report.py --improved results/r3/tracking_improvement/verification_new
```

Use new directories; the runner refuses to overwrite existing attempts. Requalify
before running if the native observer changes. The baseline and the candidate's
analysis use the same offset, footprint, speed, and intervention definitions.
"""
    (ROOT / "reports/R3_Centerline_Evaluation.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/r3/tracking_improvement/comparison"
    )
    args = parser.parse_args()
    baseline = ROOT / "results/r3/final_verification/batch_01"
    improved = args.improved.resolve()
    groups = [("Original Baseline", baseline), ("Improved Verification", improved)]
    for group, path in groups:
        if len(summaries(path)) != 5:
            raise ValueError(f"Five Measured Attempts Required: {group}")
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = []
    for group, path in groups:
        for record in summaries(path):
            comparison.append({"batch": group, "phase": "verification", **record})
    for path in sorted((ROOT / "results/r3/tracking_improvement/development").glob("*")):
        for record in summaries(path):
            comparison.append({"batch": path.name, "phase": "development", **record})
    write_csv(args.out / "all_attempts.csv", comparison)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, key, label in zip(
        axes,
        (
            "mean_absolute_lateral_offset_m",
            "std_signed_lateral_offset_m",
            "maximum_absolute_lateral_offset_m",
        ),
        ("Mean Absolute Offset (m)", "Signed-Offset SD (m)", "Maximum Absolute Offset (m)"),
    ):
        for x, (group, path) in enumerate(groups):
            values = [s[key] for s in summaries(path)]
            ax.scatter(x + np.linspace(-0.07, 0.07, 5), values, s=55, label=group)
            ax.plot([x - 0.14, x + 0.14], [np.mean(values)] * 2, color="#222222", lw=2)
        if "maximum" not in key:
            ax.axhline(0.5, color="#c83232", ls="--", label="Proposed 0.50 m Target")
        ax.set_xticks([0, 1], ["Baseline", "Improved"])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.suptitle("Independent Five-Run Batches; Every Attempt Retained", y=1.08)
    fig.savefig(args.out / "baseline_vs_improved.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    command_metrics = []
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    for ax, (group, path) in zip(axes, groups):
        experiment = json.loads((path / "experiment.json").read_text())
        start = experiment["route"]["start_station_m"]
        for i in range(1, 6):
            with (path / f"run_{i:02d}.csv").open() as stream:
                names = stream.readline().strip().split(",")
                data = np.loadtxt(stream, delimiter=",", dtype=[(name, float) for name in names])
            command_metrics.append(
                {
                    "batch": group,
                    "run": f"run_{i:02d}",
                    "total_variation_per_s": float(
                        np.abs(np.diff(data["steering"])).sum()
                        / (data["timestamp_s"][-1] - data["timestamp_s"][0])
                    ),
                    "maximum_absolute_command": float(np.abs(data["steering"]).max()),
                    "samples": len(data),
                    "mean_signed_offset_m": float(data["signed_lateral_offset_m"].mean()),
                }
            )
            data = data[:: max(1, len(data) // 3000)]
            distance = data["route_station_m"] - start
            if i == 1:
                ax.fill_between(
                    distance,
                    data["right_boundary_offset_m"],
                    data["left_boundary_offset_m"],
                    color="#ede3cf",
                    label="Authored Road Corridor",
                )
            ax.plot(distance, data["signed_lateral_offset_m"], lw=1, label=f"Run {i:02d}")
        ax.axhline(0, color="#555555", ls=":", lw=1)
        ax.set_title(group)
        ax.set_ylabel("Signed Offset (m)")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Distance Along Approved Course (m)")
    fig.savefig(args.out / "baseline_vs_improved_offset.png", dpi=300)
    plt.close(fig)
    write_csv(args.out / "command_variation.csv", command_metrics)
    (args.out / "command_variation.json").write_text(json.dumps(command_metrics, indent=2))
    table = []
    for group, path in groups:
        aggregate = json.loads((path / "aggregate.json").read_text())
        table.append({"batch": group, **aggregate})
    write_csv(args.out / "batch_comparison.csv", table)
    engineering_report(baseline, improved, args.out.resolve(), comparison)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
