"""Publication figures for actual R3 measurements only."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_results(directory: Path, runs: list[list[dict]], summaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )
    figure, axes = plt.subplots(len(runs), 1, figsize=(8, 2.0 * len(runs) + 0.7), squeeze=False)
    for ax, rows, summary in zip(axes[:, 0], runs, summaries):

        def column(key, rows=rows):
            return np.array([float(row[key]) for row in rows])

        times = column("timestamp_s")
        offset = column("signed_lateral_offset_m")
        ax.fill_between(
            times,
            column("right_boundary_offset_m"),
            column("left_boundary_offset_m"),
            color="#e6edf2",
            label="Exported Road Corridor",
        )
        ax.plot(
            times,
            column("vehicle_center_lower_limit_m"),
            color="#8b6530",
            ls="--",
            lw=0.8,
            label="Width-Adjusted Local Limits",
        )
        ax.plot(times, column("vehicle_center_upper_limit_m"), color="#8b6530", ls="--", lw=0.8)
        ax.axhline(0, color="#697077", lw=0.8, ls=":")
        ax.plot(times, offset, color="#126a9a", lw=1.25, label="Signed Offset")
        outside = column("road_departure").astype(bool)
        ax.scatter(
            times[outside],
            offset[outside],
            color="#bb3131",
            s=9,
            zorder=4,
            label="Footprint Departure",
        )
        ax.set_ylabel("Lateral Offset (m)")
        ax.set_title(summary["run"].replace("_", " ").title(), loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Simulation Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), fontsize=8, ncol=2
    )
    figure.suptitle(
        "Figure 1: Lateral Offset Over Time\nProvisional Route; Graphics-Step Samples", fontsize=12
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(directory / "lateral_offset_over_time.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = [summary["run"].replace("_", " ").title() for summary in summaries]
    x = np.arange(len(labels))
    means = [summary["mean_absolute_lateral_offset_m"] for summary in summaries]
    absolute_std = [summary["std_absolute_lateral_offset_m"] for summary in summaries]
    axes[0].bar(x, means, color="#126a9a", width=0.65)
    # Asymmetric clipping keeps the visual band nonnegative; the full numerical
    # absolute-error SD is preserved in the CSV and the upper error bar.
    axes[0].errorbar(
        x,
        means,
        yerr=[np.minimum(means, absolute_std), absolute_std],
        fmt="none",
        color="#263746",
        capsize=4,
        lw=1.2,
    )
    axes[0].set_title("Mean Absolute Offset", fontsize=11)
    axes[0].set_ylabel("Mean Absolute Offset (m)")
    axes[1].bar(
        x,
        [summary["std_signed_lateral_offset_m"] for summary in summaries],
        color="#467b65",
        width=0.65,
    )
    axes[1].set_title("Standard Deviation Of Signed Offset", fontsize=11)
    axes[1].set_ylabel("Population Standard Deviation (m)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    figure.suptitle("Figure 2: Lateral Offset Across Runs (Provisional)", fontsize=12)
    figure.text(
        0.5,
        0.01,
        "Error Bars: ±1 Population SD Of Absolute Error (Lower Bars Clipped At Zero); "
        "Not Confidence Intervals",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(directory / "lateral_offset_comparison.png", bbox_inches="tight")
    plt.close(figure)
