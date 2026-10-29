#!/usr/bin/env python3
"""Write The R3 Engineering Report From A Complete Measured Five-Run Batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value, digits=4) -> str:
    if value is None:
        return "Not Measured"
    return f"{value:.{digits}f}"


def verdict(value, integrity=True) -> str:
    if not integrity:
        return "Unverified"
    if value:
        return "Pass"
    return "Fail"


def write_report(batch: Path, report: Path) -> None:
    summaries = [json.loads((batch / f"run_{i:02d}.summary.json").read_text()) for i in range(1, 6)]
    manifests = [json.loads((batch / f"run_{i:02d}.json").read_text()) for i in range(1, 6)]
    experiment = json.loads((batch / "experiment.json").read_text())
    aggregate = json.loads((batch / "aggregate.json").read_text())
    audit = json.loads((batch / "data_audit.json").read_text())
    validation = json.loads((batch / "validation.json").read_text())
    if not audit["all_five_integrity_checks_passed"]:
        raise ValueError("The Recorded Data Audit Did Not Pass")
    route = experiment["route"]
    if any(
        m["source_hashes"] != experiment["source_hashes"] or m["route"] != route for m in manifests
    ):
        raise ValueError("Run Manifests Do Not Match The Frozen Batch")
    ref = "../" + batch.relative_to(ROOT).as_posix()
    status = "Yellow"
    confirmed_failure = any(
        not s["route_completed"]
        or s["road_departures"] > 0
        or s["manual_interventions"] > 0
        or not s["speed_compliance"]
        for s in summaries
    )
    if confirmed_failure:
        status = "Red"
    proposed_passes = sum(s["all_proposed_test_conditions"] for s in summaries)
    completed = sum(s["route_completed"] for s in summaries)
    integrity_count = sum(s["physics_recording_integrity"] for s in summaries)
    quantitative = []
    assessment = []
    reasons = []
    for i, s in enumerate(summaries, 1):
        completed_text = "No"
        if s["route_completed"]:
            completed_text = "Yes"
        quantitative.append(
            f"| {i:02d} | {number(s['mean_absolute_lateral_offset_m'])} | "
            f"{number(s['std_signed_lateral_offset_m'])} | {number(s['maximum_absolute_lateral_offset_m'])} | "
            f"{s['road_departures']} | {s['manual_interventions']} | {number(s['maximum_speed_mph'])} | "
            f"{completed_text} | {number(s['duration_s'], 3)} |"
        )
        good = s["physics_recording_integrity"]
        assessment.append(
            f"| {i:02d} | {verdict(s['proposed_mean_target'], good)} | "
            f"{verdict(s['proposed_signed_sd_target'], good)} | "
            f"{verdict(s['zero_authored_corridor_departures'], good)} | "
            f"{verdict(s['zero_manual_inputs_verified'], good)} | "
            f"{verdict(s['speed_compliance'], good)} | {verdict(s['route_completed'], good)} | "
            f"{verdict(s['all_proposed_test_conditions'], good)} |"
        )
        reasons.append(f"| {i:02d} | {s['samples']:,} | {s['end_reason']} | {verdict(good)} |")
    mean_passes = sum(s["proposed_mean_target"] for s in summaries)
    sd_passes = sum(s["proposed_signed_sd_target"] for s in summaries)
    text = f"""# R3: Closeness to Centerline (Simulation Only)

**Status:** {status}. The approved-bends batch completed {completed}/5 routes and
{proposed_passes}/5 attempts met all proposed test conditions. Mean absolute offset
met the proposed 0.50 m limit in {mean_passes}/5 runs; signed-offset SD met it in
{sd_passes}/5. The centering targets await mentor approval, and the allowable
physical trail boundary still requires confirmation. No final R3 compliance is claimed.

## Acceptance Criteria

| Metric | Criterion | Standing |
| --- | --- | --- |
| Mean Absolute Lateral Offset | At Most 0.50 m Per Run | Proposed Engineering Target; Mentor Approval Pending |
| Population SD Of Signed Offset | At Most 0.50 m Per Run | Proposed Engineering Target; Mentor Approval Pending |
| Road Departures | Zero | Confirmed Requirement; Physical Boundary Definition Pending |
| Manual Interventions | Zero | Confirmed Requirement |
| Maximum Actual Speed | At Most 10 mph | Confirmed Requirement |
| Completion | Entire Approved A-To-B Traversal | User Approved Open Traversal Through The Highlighted Bends |
| Conditions | BeamNG, Hopper, Unmarked Dirt Road, Sunny Daytime, Fully Autonomous | Fixed Test Conditions |

No official sponsor-approved centering threshold is asserted. No pass/fail limit
is invented for maximum absolute offset, which is reported diagnostically. Speed
assessment adds no overspeed tolerance. A completed simulation does not establish
centering compliance or resolve unverified boundary/intervention evidence.

## Current Implementation Status

The existing RGB semantic-perception pipeline, baseline planner, Stanley steering
with curvature feedforward, and longitudinal control remain unchanged. Wheel/gyro
odometry remains the controller's motion source. Evaluation ground truth is never
fed into perception, planning, or control. No controller tuning was performed in
the scored batch.

The verification adapter adds native physics-rate pose and footprint recording,
input-source exclusivity, deliberate-intervention qualification, independent road
projection/containment, and source/configuration freeze checks. All five attempts,
including any failed criterion or stop, are retained. Tests stop automatically on
an observed envelope crossing, intervention/authority fault, excessive speed, or
an existing application safety condition. A stop before B is incomplete.

The previous five 16.596 m straight-route runs remain **preliminary** at the top
level of `results/r3/` and are excluded from every new statistic below. Their report
is preserved in [The Preliminary Evaluation](../results/r3/Preliminary_R3_Centerline_Evaluation.md).
Earlier exploratory records and qualification failures are also preserved.

## Experimental Setup And Methodology

The initial screenshot described a closed loop, but the actual dirt road is open
and serpentine. The user subsequently approved an open traversal through both
highlighted bends on road `23528.0`, persistent ID
`ccc97201-4b41-4e28-8f93-afd97e6c2d41`, material `m_dirt_road_01`. The nearby invisible
navigation road is not used as the physical corridor. No connector was invented.

| Setup Item | Exact Value |
| --- | --- |
| Map And Vehicle | `automation_test_track`; Hopper `/vehicles/hopper/lxt6_M.pc` |
| Simulator Version | `{manifests[0]["beamng_version"]}` |
| Package Versions | {"; ".join(f"{name} {version}" for name, version in experiment["versions"].items())} |
| Selected Nodes | Original Exported Nodes {route["start_node_index"]} Through {route["finish_node_index"]}, Increasing Order |
| Point A XYZ (m) | `{route["point_a_xyz_m"]}` |
| Point B XYZ (m) | `{route["point_b_xyz_m"]}` |
| A/B Stations On Complete Road | {route["start_station_m"]:.12f} m / {route["finish_station_m"]:.12f} m |
| Horizontal Course Length | **{route["route_length_m"]:.12f} m** |
| Initial Spawn XYZ (m) | `{route["spawn_position"]}` |
| Initial Quaternion XYZW | `{route["spawn_rotation"]}` |
| Direction | West Around The Western Bend, East Through The Middle, Around The Eastern Return Bend, Then West To B |
| Gates | A Within 1.50 m XY And Forward/Tangent Cosine At Least 0.95; B Station Reached Within 2.00 m XY |
| Environment | Sunny-Noon Preset; Frozen Time Of Day 0.0, Recorded As 12:00:00 |
| Initial Condition | Normal Clinging/Settling; Two Parked Inference Warmups; Pipeline And Local Odometry Reset Before Scoring |
| Seed | Python, NumPy, Torch And CUDA: 0 Each Run; Simulator-Wide RNG Not Controlled |
| Nominal Driving Target | Unchanged Default 5.5 mph; Requirement Ceiling 10 mph |
| Native Physics Interval | Approximately 0.000500000024 s, Or 2,000 Hz |
| Vehicle Envelope | Calibrated Visible-Body OOBB Plus All 854 Current JBeam Nodes; 0.05 m Margin |

The 210-cross-section [Road Snapshot](../results/r3/final_verification/approved_route_geometry.json)
preserves actual BeamNG left/middle/right world XYZ geometry. Every live run checks
its edges against this snapshot. The complete object is approximately 661.894 m;
only the documented 332.544 m course is scored. The screenshot's approximate
coordinates served only to locate the road. The default Spawn 0 was a locator;
the new runs start at A, with spawn Z placed 0.50 m above its authored centerline
before normal terrain clinging. The observed settled position is saved per run.

The authored DecalRoad ribbon is the explicit test boundary. Its dirt appearance
at A was inspected, but independent confirmation of the permissible physical
trail edge remains outstanding. Numeric results below establish authored-corridor
performance, not a substitute physical boundary definition.

### Manual Intervention Qualification

The observer allows only the private autonomy input source for steering, throttle,
brake, parking brake, and clutch, using BeamNG's native source whitelist. Wrappers
record/reject supported keyboard, gamepad, direct-event, gear/gearbox-mode,
parking-toggle, ignition, and available four-wheel-drive requests. Vehicle resets
are flagged. Source permissions, raw command values, and wrapper identity are
checked at every physics step. Missing telemetry or a guard fault prevents a
zero-intervention verification.

Before this batch, eight deliberate native-input challenges were detected and
blocked, generating nine rejected nonzero command events. One external request
contained two channels. A separate clean motion interval recorded zero interventions
and zero authority faults. This qualification exercised the actual binding entry
points, not physical gamepad hardware. Camera/view controls are excluded. Neutral
external updates are blocked without being counted as nonzero takeover commands.
The scope is ordinary simulator driving inputs, not an adversarial operator
executing arbitrary Lua or changing process memory.

[Qualification Attempt 03](../results/r3/final_verification/qualification/attempt_03/qualification.json)
matches the observer hash required by the scored runner. Prior qualification
attempts remain adjacent. Only the matching final qualification was used to
authorize this batch. Input entry points are described in
[BeamNG Input Actions](https://docs.beamng.com/modding/input/actions/).

### Physics Footprint, Offset, And Statistics

Every native vehicle-Lua physics callback records position, forward/up vectors,
actual world velocity, six envelope bounds, commands, sequence, time, intervention
count, and authority faults. Current extents of all 854 JBeam nodes are recomputed
each step and combined with the calibrated visible-body OOBB. This includes wheel
articulation. A separate moving-pose comparison found no excess beyond the
configured 0.05 m envelope margin. Native vehicle physics-frequency execution is
documented in [BeamNG Virtual Machines](https://www.documentation.beamng.com/modding/programming/virtualmachines/).

All eight world envelope corners are reconstructed, and their entire convex XY
projection must lie in the authored polygon. This detects concave boundary intrusions
that a corner-only test misses. Envelope crossings are conservative safety failures,
not tire-contact observations. The method observes every physics integration state;
it does not claim an analytical continuous-time sweep between states. Containment
of the conservative envelope supports containment of the covered vehicle geometry
within the authored ribbon. Physical trail-edge approval remains separate.

Centering uses the **actual BeamNG vehicle reference position** in world XY metres.
For each eligible segment, `u = clip(dot(p-a,b-a)/dot(b-a,b-a), 0, 1)` gives the
nearest point; the shortest horizontal distance supplies the magnitude. Sign is
positive left along the course. Z is interpolated for audit and a 3 m elevation
gate. Station continuity is checked with a 12 m window. No predicted geometry or
odometry position is used as truth, and longitudinal overrun is not labeled lateral
error. The reference position is not necessarily the body centroid or center of
mass; formal reference-point agreement remains part of protocol review.

Mean absolute error is `mean(abs(e))`, signed SD is `std(e, ddof=0)`, and
absolute-error SD is `std(abs(e), ddof=0)`. Maximum absolute offset is reported
separately. Departures are contiguous outside intervals. Actual speed is the
XYZ velocity magnitude divided by 0.44704. The scored interval ends at the
recorded B gate or unsuccessful stop; post-finish raw observations are retained.
Fixed-rate physics samples give effectively duration-weighted pooled statistics.
Samples are temporally correlated; five runs are the experimental repeats.

Complete implementation details are in [The Methodology Appendix](R3_Verification_Methodology.md).
The [Batch Manifest]({ref}/experiment.json) records resolved settings, versions,
model/source and geometry hashes, and qualification provenance. Frozen hashes are
checked before every attempt. Simulator/rendering timing is not bitwise deterministic.

## Results

**Table 1: Actual Physics-Rate Simulation Performance**

| Run | Mean Absolute Offset (m) | Signed SD (m) | Maximum Absolute Offset (m) | Envelope Departures | Manual Events | Maximum Speed (mph) | Completed | Duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
{chr(10).join(quantitative)}

**Table 2: Assessments Against Proposed Targets And Recorded Test Conditions**

| Run | Mean At Most 0.50 m | Signed SD At Most 0.50 m | Zero Authored-Corridor Crossings | Zero Interventions | Speed At Most 10 mph | Full Route | All Proposed Conditions |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(assessment)}

The centering assessments in Table 2 are provisional engineering assessments.
They are not official sponsor-approved compliance declarations. The zero-crossing
assessment uses the authored road ribbon and conservative vehicle envelope.

| Aggregate Metric | Result |
| --- | ---: |
| Completed Routes | {completed}/5 |
| Attempts Meeting All Proposed Conditions | {proposed_passes}/5 |
| Scored Physics Samples | {aggregate["samples"]:,} |
| Pooled Mean Absolute Offset | {number(aggregate["pooled_mean_absolute_lateral_offset_m"])} m |
| Pooled Signed-Offset SD | {number(aggregate["pooled_std_signed_lateral_offset_m"])} m |
| Pooled Absolute-Error SD | {number(aggregate["pooled_std_absolute_lateral_offset_m"])} m |
| Maximum Absolute Offset | {number(aggregate["maximum_absolute_lateral_offset_m"])} m |
| Envelope Departure Episodes | {aggregate["road_departures"]} |
| Rejected External Driving Command/Reset Events | {aggregate["manual_interventions"]} |
| Maximum Actual Speed | {number(aggregate["maximum_speed_mph"])} mph |
| Runs With Valid Native Physics/Authority Records | {integrity_count}/5 |

Signed-offset SD and absolute-error SD measure different distributions and are
not interchangeable. Pooled signed SD includes between-run bias differences;
it is not the mean of the five individual SDs. Figure 2's error bars show one
population SD of absolute error, not confidence intervals or signed SD.

![Figure 1: Signed Lateral Offset Over Time]({ref}/lateral_offset_over_time.png)

![Figure 2: Mean Absolute Offset And Signed SD Against Proposed Targets]({ref}/lateral_offset_comparison.png)

![Table 1: Simulation Performance Summary]({ref}/performance_summary.png)

![Figure 3: Approved Route, A, B, And Measured Trajectories]({ref}/route_map.png)

Full-precision values are in [The Summary CSV]({ref}/summary.csv) and
[Aggregate JSON]({ref}/aggregate.json). Each `run_NN.csv` contains every scored
physics measurement, including signed and absolute errors. Native `.physics.csv`
records preserve the underlying pose, frame, envelope, velocity, commands, and
integrity counters. Qualification interventions are not pooled with scored runs.
Lossless `.physics.csv.gz` archives are retained for transfer and version control;
each decompressed archive was SHA-256 checked against its locally retained original.

The [Data Integrity Audit]({ref}/data_audit.json) passed all five runs: no native
sequence gaps, authority faults, unexplained pose discontinuities, or recorded
vehicle damage were found. Independent scalar projection and footprint checks
matched the vectorized scorer at 41 samples per run, including each maximum-error
sample. Source, route, geometry, measurement, and archive hashes are recorded.

| Run | Scored Samples | Termination Reason | Physics/Authority Integrity |
| --- | ---: | --- | --- |
{chr(10).join(reasons)}

## Results Interpretation

The recorded outcome is {completed}/5 complete approved traversals and
{proposed_passes}/5 attempts satisfying all proposed conditions. Completion and
boundary/input safety must be assessed separately from centering. Only
{mean_passes}/5 runs met the proposed mean-offset limit and {sd_passes}/5 met the
proposed signed-SD limit. These results therefore do not justify declaring the
centering requirement passed simply because the vehicle reached B.

The aggregate distinguishes signed spread from variation in absolute error. The
maximum deviation identifies excursions that a mean alone would conceal. The route
map and time histories support spatial and temporal review of those excursions.
No failed attempt or criterion was removed, and no straight-route data were
substituted for this curved course.

## Remaining Challenges And Limitations

The two 0.50 m limits await mentor approval. Exact test gates, the vehicle reference
point, and the allowable physical dirt-road edge should be confirmed in the final
protocol. Authored-ribbon containment does not establish a separately defined
physical trail-edge criterion. The 0.05 m envelope margin makes crossing detection
conservative and is not an invented centering target.

The data establish performance on this particular course, speed range, vehicle,
and warmed initial condition. They do not certify cold-start reliability, every
map, or every terrain surface. Flat-ground RGB projection and perception/planning
errors remain possible contributors, but these measurements alone do not isolate
their causal contribution. The observer adds runtime load; simulator-wide RNG,
render timing, and inference timing are not controlled deterministically.

The native guard covers ordinary driving input APIs and checks authority every
physics step. It is not a process-security boundary. The envelope includes all
current physics nodes and the calibrated nominal body envelope; its interpretation
and physical boundary definition should be reviewed together. Horizontal metrics
do not represent terrain-surface distance on steep banks. Millions of correlated
physics samples do not replace additional independent runs and scenarios.

## Next Steps

1. Obtain mentor approval or revision of the proposed centering limits, and sign off
   the exact gates, reference position, and allowable physical road boundary.
2. Review the measured centering excursions using the time histories and map. Any
   perception, planner, or controller improvement belongs in separate development
   runs, followed by a new frozen complete five-run batch. Preserve this batch.
3. Retain and independently audit the native physics data, geometry, source hashes,
   and deliberate-intervention qualification. Requalify if the observer changes.
4. Claim official compliance only when the applicable approved criteria are met
   with credible whole-route evidence. Until then, the status remains **{status}**.

## Reproduction And Evidence Preservation

The original short-route runs and preliminary report remain separate from
`results/r3/final_verification/batch_01/`. All five new attempt manifests, native
records, derived CSVs, summaries, and figures are retained. Existing evidence is
not overwritten by the run command.

The [Software Validation Record]({ref}/validation.json) records
{validation["pytest"]["passed"]} passing tests, {validation["pytest"]["skipped"]} skips,
and {validation["pytest"]["coverage_percent"]:.2f}% coverage. Repository lint,
formatting, the synthetic S-bend regression, and wheel/source-package builds
passed. Synthetic checks validate software behavior; their values are excluded
from the measured BeamNG results.

```powershell
python scripts/r3_verify.py qualify --out results/r3/final_verification/qualification_new
python scripts/r3_verify.py run --runs 5 --qualification results/r3/final_verification/qualification_new/qualification.json --out results/r3/final_verification/batch_new
python scripts/r3_verify.py analyze --out results/r3/final_verification/batch_new
```
"""
    report.write_text(text, encoding="utf-8")
    print(f"R3 Report Written: {report}; Status: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, default=ROOT / "results/r3/final_verification/batch_01"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports/R3_Centerline_Evaluation.md")
    args = parser.parse_args()
    write_report(args.batch, args.report)


if __name__ == "__main__":
    main()
