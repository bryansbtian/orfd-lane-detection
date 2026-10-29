"""Figures For Measured Attempts On The Approved Dirt-Road Bends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _measurements(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        names = stream.readline().strip().split(",")
        data = np.loadtxt(stream, delimiter=",", dtype=[(name, float) for name in names])
    return np.atleast_1d(data)


def make_figures(directory: Path, route: dict, summaries: list[dict], geometry_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
    recordings = [_measurements(directory / f"{s['run']}.csv") for s in summaries]
    figure, axes = plt.subplots(
        len(recordings), 1, figsize=(9, 2 * len(recordings) + 1.2), squeeze=False
    )
    for ax, data, summary in zip(axes[:, 0], recordings, summaries):
        stride = max(1, len(data) // 5000)
        sampled = data[::stride]
        t = sampled["timestamp_s"]
        ax.fill_between(
            t,
            sampled["right_boundary_offset_m"],
            sampled["left_boundary_offset_m"],
            color="#ede3cf",
            label="Authored Road Corridor",
        )
        ax.plot(
            t,
            sampled["vehicle_center_lower_limit_m"],
            color="#9b702a",
            ls="--",
            lw=0.8,
            label="Envelope-Adjusted Local Limits",
        )
        ax.plot(t, sampled["vehicle_center_upper_limit_m"], color="#9b702a", ls="--", lw=0.8)
        ax.axhline(0, color="#666666", ls=":", lw=0.8)
        ax.plot(
            t, sampled["signed_lateral_offset_m"], color="#14759c", lw=1.1, label="Signed Offset"
        )
        outside = data["road_departure"] > 0
        ax.scatter(
            data["timestamp_s"][outside],
            data["signed_lateral_offset_m"][outside],
            color="#c43b40",
            s=5,
            zorder=5,
            label="Envelope Departure",
        )
        ax.set_title(
            summary["run"].replace("_", " ").title() + ": " + summary["end_reason"],
            loc="left",
            fontsize=9,
        )
        ax.set_ylabel("Signed Offset (m)")
        ax.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Simulation Time (s)")
    figure.suptitle("R3: Approved Dirt-Road Bends\nNative Physics-Step Measurements", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, fontsize=8
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(directory / "lateral_offset_over_time.png", bbox_inches="tight")
    plt.close(figure)

    labels = [s["run"].replace("_", " ").title() for s in summaries]
    x = np.arange(len(summaries))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    means = np.array([s["mean_absolute_lateral_offset_m"] for s in summaries], dtype=float)
    absolute_sd = np.array([s["std_absolute_lateral_offset_m"] for s in summaries], dtype=float)
    axes[0].bar(x, means, color="#14759c")
    axes[0].errorbar(
        x,
        means,
        yerr=[np.minimum(means, absolute_sd), absolute_sd],
        fmt="none",
        color="#253847",
        capsize=4,
    )
    axes[1].bar(x, [s["std_signed_lateral_offset_m"] for s in summaries], color="#487f67")
    axes[0].set_ylabel("Mean Absolute Offset (m)")
    axes[1].set_ylabel("SD Of Signed Offset (m)")
    for ax in axes:
        ax.axhline(0.5, color="#bd3838", ls="--", lw=1.3, label="Proposed Target: 0.50 m")
        ax.set_xticks(x, labels, rotation=20)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    figure.suptitle(
        "Centering Against Proposed Engineering Targets\nMentor Approval Pending", fontsize=12
    )
    figure.text(
        0.5,
        0.015,
        "Error Bars: One Population SD Of Absolute Error; Not Confidence Intervals",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.89))
    figure.savefig(directory / "lateral_offset_comparison.png", bbox_inches="tight")
    plt.close(figure)

    rows = []
    for s in summaries:
        status = "Fail"
        if s["all_proposed_test_conditions"]:
            status = "Pass (Proposed)"
        completed = "No"
        if s["route_completed"]:
            completed = "Yes"
        rows.append(
            [
                s["run"],
                f"{s['mean_absolute_lateral_offset_m']:.3f}",
                f"{s['std_signed_lateral_offset_m']:.3f}",
                f"{s['maximum_absolute_lateral_offset_m']:.3f}",
                str(s["road_departures"]),
                str(s["manual_interventions"]),
                f"{s['maximum_speed_mph']:.2f}",
                completed,
                status,
            ]
        )
    figure, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Run",
            "Mean |e|\n(m)",
            "SD(e)\n(m)",
            "Max |e|\n(m)",
            "Departures",
            "Manual\nInputs",
            "Max Speed\n(mph)",
            "Complete",
            "All Proposed\nConditions",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)
    figure.suptitle(
        "Simulation Performance Summary | 0.50 m Centering Targets Await Mentor Approval",
        fontsize=12,
    )
    figure.savefig(directory / "performance_summary.png", bbox_inches="tight")
    plt.close(figure)

    snapshot = json.loads(geometry_path.read_text(encoding="utf-8"))
    edges = snapshot["road"]["edges"]
    center = np.array([e["middle"] for e in edges])
    left = np.array([e["left"] for e in edges])
    right = np.array([e["right"] for e in edges])
    polygon = np.vstack((left[:, :2], right[::-1, :2]))
    a, b = route["start_node_index"], route["finish_node_index"]
    figure, ax = plt.subplots(figsize=(11, 7))
    ax.fill(polygon[:, 0], polygon[:, 1], color="#ede3cf", label="Authored Road Corridor")
    ax.plot(center[:, 0], center[:, 1], color="#9b702a", ls="--", lw=1, label="Authored Centerline")
    ax.plot(
        center[a : b + 1, 0],
        center[a : b + 1, 1],
        color="#24a773",
        lw=3,
        label="Approved A-To-B Traversal",
    )
    for data, summary in zip(recordings, summaries):
        stride = max(1, len(data) // 10000)
        ax.plot(
            data["vehicle_x_m"][::stride],
            data["vehicle_y_m"][::stride],
            lw=1.2,
            label=summary["run"].replace("_", " ").title(),
        )
    for index, label in ((a, "A: Start"), (b, "B: Finish")):
        ax.scatter(center[index, 0], center[index, 1], s=65, color="#122e44", zorder=5)
        ax.annotate(
            label, center[index, :2], xytext=(8, 12), textcoords="offset points", fontsize=11
        )
    for index in (40, 75, 105, 132):
        ax.annotate(
            "",
            xy=center[index + 2, :2],
            xytext=center[index, :2],
            arrowprops={"arrowstyle": "->", "color": "#14754f", "lw": 1.5},
        )
    selected = center[a : b + 1, :2]
    ax.set_xlim(selected[:, 0].min() - 12, selected[:, 0].max() + 12)
    ax.set_ylim(selected[:, 1].min() - 12, selected[:, 1].max() + 12)
    ax.set_aspect("equal")
    ax.set_xlabel("BeamNG World X (m)")
    ax.set_ylabel("BeamNG World Y (m)")
    ax.set_title(
        f"Approved Open Traversal Through The Highlighted Bends\n{route['route_length_m']:.3f} m; Actual Recorded Vehicle Trajectories"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(directory / "route_map.png", bbox_inches="tight")
    plt.close(figure)
    e = np.concatenate([data["signed_lateral_offset_m"] for data in recordings])
    aggregate = {
        "measured_attempts": len(summaries),
        "samples": len(e),
        "completed_runs": sum(s["route_completed"] for s in summaries),
        "runs_meeting_all_proposed_conditions": sum(
            s["all_proposed_test_conditions"] for s in summaries
        ),
        "pooled_mean_absolute_lateral_offset_m": float(np.mean(np.abs(e))),
        "pooled_std_signed_lateral_offset_m": float(np.std(e, ddof=0)),
        "pooled_std_absolute_lateral_offset_m": float(np.std(np.abs(e), ddof=0)),
        "maximum_absolute_lateral_offset_m": float(np.max(np.abs(e))),
        "road_departures": sum(s["road_departures"] for s in summaries),
        "manual_interventions": sum(s["manual_interventions"] for s in summaries),
        "maximum_speed_mph": max(s["maximum_speed_mph"] for s in summaries),
    }
    (directory / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
