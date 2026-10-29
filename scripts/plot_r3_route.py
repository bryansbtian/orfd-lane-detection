#!/usr/bin/env python3
"""Draw The Provisional R3 Route From Saved BeamNG Geometry."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry


def draw_route(route_path: Path, survey_path: Path, output: Path) -> None:
    route = yaml.safe_load(route_path.read_text(encoding="utf-8"))
    snapshot = json.loads((route_path.parent / route["geometry_file"]).read_text(encoding="utf-8"))
    edges = snapshot["road"]["edges"]
    if route.get("reverse", False):
        edges = list(reversed(edges))
    geometry = RoadGeometry(edges)
    station = np.linspace(route["start_station_m"], route["finish_station_m"], 100)
    path = np.column_stack(
        [np.interp(station, geometry.station, geometry.center[:, axis]) for axis in (0, 1)]
    )
    a = np.asarray(route["point_a_xyz_m"][:2])
    b = np.asarray(route["point_b_xyz_m"][:2])
    spawn = np.asarray(route["spawn_position"][:2])
    midpoint = (a + b) / 2
    detail = (min(a[0], b[0]) - 6, max(a[0], b[0]) + 6, midpoint[1] - 9, midpoint[1] + 9)
    context = (midpoint[0] - 60, midpoint[0] + 70, midpoint[1] - 55, midpoint[1] + 80)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.2]})
    overview, closeup = axes
    if survey_path.is_file():
        survey = json.loads(survey_path.read_text(encoding="utf-8"))
        first = True
        for road in survey["roads"].values():
            if "dirt_road" not in str(road.get("material", "")) or not road.get("edges"):
                continue
            center = np.array([edge["middle"][:2] for edge in road["edges"]])
            if (
                center[:, 0].max() < context[0]
                or center[:, 0].min() > context[1]
                or center[:, 1].max() < context[2]
                or center[:, 1].min() > context[3]
            ):
                continue
            label = None
            if first:
                label = "Other Dirt Road Geometry"
                first = False
            overview.plot(center[:, 0], center[:, 1], color="#b9ad99", lw=2.5, label=label)

    for ax in axes:
        ax.fill(
            geometry.polygon[:, 0],
            geometry.polygon[:, 1],
            color="#dfcba5",
            alpha=0.7,
            label="Exported Road Corridor",
        )
        for boundary in (geometry.left, geometry.right):
            ax.plot(boundary[:, 0], boundary[:, 1], color="#8c7245", lw=0.9)
        ax.plot(
            geometry.center[:, 0],
            geometry.center[:, 1],
            color="#8c7245",
            ls="--",
            lw=1,
            label="Surveyed Centerline",
        )
        ax.plot(path[:, 0], path[:, 1], color="#087ea4", lw=3, label="Provisional A To B Route")
        ax.scatter(*a, color="#087ea4", edgecolor="white", s=85, zorder=5)
        ax.scatter(*b, marker="X", color="#c43b40", s=90, zorder=5)
        ax.scatter(*spawn, marker="<", color="#202c39", s=65, zorder=6, label="Configured Spawn")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("BeamNG World X (m)")
        ax.set_ylabel("BeamNG World Y (m)")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    overview.set_xlim(context[:2])
    overview.set_ylim(context[2:])
    overview.set_title("Local Map Context", loc="left", fontweight="bold")
    overview.add_patch(
        Rectangle(
            (detail[0], detail[2]),
            detail[1] - detail[0],
            detail[3] - detail[2],
            fill=False,
            edgecolor="#087ea4",
            lw=1.2,
            ls=":",
        )
    )
    overview.annotate(
        "Detail At Right",
        xy=(detail[0], detail[3]),
        xytext=(-12, 20),
        textcoords="offset points",
        fontsize=9,
        color="#087ea4",
    )

    closeup.set_xlim(detail[:2])
    closeup.set_ylim(detail[2:])
    closeup.set_title("Route Detail: Westbound", loc="left", fontweight="bold")
    closeup.annotate(
        "A: Start Reference",
        xy=a,
        xytext=(-85, 58),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#087ea4"},
        color="#087ea4",
    )
    closeup.annotate(
        "B: Finish",
        xy=b,
        xytext=(-25, 60),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#c43b40"},
        color="#c43b40",
    )
    closeup.annotate(
        "Vehicle Spawn",
        xy=spawn,
        xytext=(-75, -65),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#202c39"},
    )
    closeup.annotate(
        "",
        xy=path[62],
        xytext=path[40],
        arrowprops={"arrowstyle": "-|>", "color": "#087ea4", "lw": 2, "mutation_scale": 20},
    )
    trajectory_file = ROOT / "results/r3/run_01.csv"
    if trajectory_file.is_file() and route_path.resolve() == (ROOT / "configs/r3_provisional.yaml"):
        with trajectory_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        closeup.plot(
            [float(row["vehicle_x_m"]) for row in rows],
            [float(row["vehicle_y_m"]) for row in rows],
            color="#8751a5",
            lw=1.5,
            label="Measured Vehicle Path: Run 01",
        )
    distance = route["finish_station_m"] - route["start_station_m"]
    closeup.text(
        0.03,
        0.04,
        f"Route Length: {distance:.3f} m\n+X Right, +Y Up; Equal Scale",
        transform=closeup.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d7d7d7", "alpha": 0.95},
    )
    handles = {}
    for ax in axes:
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists))
    figure.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(
        "R3 Provisional Route Map\nAutomation Test Track | Saved BeamNG World Geometry",
        fontsize=15,
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        "Provisional Route. Exported DecalRoad Edges Have Not Been Validated As Physical "
        "Trail Boundaries.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.18, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(f"Route Map Written: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, default=ROOT / "configs/r3_provisional.yaml")
    parser.add_argument("--survey", type=Path, default=ROOT / "output/r3/road_survey.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/r3/route_map.png")
    args = parser.parse_args()
    draw_route(args.route, args.survey, args.out)


if __name__ == "__main__":
    main()
