"""Compare two runs: postprocessing disabled vs enabled.

Usage:
    python scripts/plot_comparison.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv


def load_log(path: str) -> dict:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["drive_mode"] == "AUTO":
                rows.append(row)
    if not rows:
        return {}
    t      = [float(r["time_s"])            for r in rows]
    t0     = t[0]
    t      = [x - t0                        for x in t]
    steer  = [float(r["steering"])          for r in rows]
    speed  = [float(r["speed_mps"])         for r in rows]
    kalman = [r["kalman_active"] == "True"  for r in rows]
    return dict(t=t, steer=steer, speed=speed, kalman=kalman)


LOG_DIR = Path(__file__).parent.parent / "simulator_log"
logs = sorted(LOG_DIR.glob("*.csv"))[-2:]  # two most recent

if len(logs) < 2:
    sys.exit("Need at least 2 log files in simulator_log/")

# Assume older = no postprocessing, newer = with postprocessing
raw_data  = load_log(str(logs[0]))   # older (no postprocessing)
post_data = load_log(str(logs[1]))   # newer (with postprocessing)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
fig.suptitle("Effect of Temporal Postprocessing on Driving Stability", fontsize=14, fontweight="bold")

COLOR_RAW  = "#e05c5c"
COLOR_POST = "#5cb85c"
ALPHA_FILL = 0.15

for ax, data, label, color in [
    (axes[0], raw_data,  "No postprocessing (EMA disabled)", COLOR_RAW),
    (axes[1], post_data, "With postprocessing (EMA + morphology)", COLOR_POST),
]:
    t     = data["t"]
    steer = data["steer"]
    speed = data["speed"]
    kalman = data["kalman"]

    # Shade Kalman dropout regions
    in_dropout = False
    d_start = 0
    for i, k in enumerate(kalman):
        if k and not in_dropout:
            d_start = t[i]
            in_dropout = True
        elif not k and in_dropout:
            ax.axvspan(d_start, t[i], color="orange", alpha=0.25, label="Mask dropout" if d_start == t[next(j for j,x in enumerate(kalman) if x)] else "")
            in_dropout = False
    if in_dropout:
        ax.axvspan(d_start, t[-1], color="orange", alpha=0.25)

    ax.plot(t, steer, color=color, linewidth=1.2, label="Steering")
    ax.fill_between(t, steer, 0, color=color, alpha=ALPHA_FILL)
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)

    # Mark end of run (crash)
    ax.axvline(t[-1], color="red", linewidth=2, linestyle="--", label=f"End ({len(t)} frames)")

    stdev = float(np.std(steer))
    ax.set_title(f"{label}   |   {len(t)} frames survived   |   steering σ = {stdev:.3f}", fontsize=11)
    ax.set_ylabel("Steering [-1, 1]")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(0, max(t) + 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

axes[1].set_xlabel("Time (s)")
axes[1].xaxis.label.set_color("white")

fig.patch.set_facecolor("#12121f")
plt.tight_layout()

out = LOG_DIR / "postprocessing_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.show()
