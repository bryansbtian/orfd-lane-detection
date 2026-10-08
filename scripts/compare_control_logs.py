"""Compare live control CSVs. Errors are perception-relative, not road ground truth."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least two samples")

    def col(name):
        return np.array([float(row[name]) for row in rows])

    t, steer = col("t"), col("steer")
    active = steer[np.abs(steer) > 0.03]
    return {
        "file": str(path),
        "frames": len(rows),
        "duration_s": float(t[-1]),
        "distance_m": float(np.hypot(np.diff(col("x")), np.diff(col("y"))).sum()),
        "mean_cte_m": float(np.abs(col("cte")).mean()),
        "max_cte_m": float(np.abs(col("cte")).max()),
        "mean_heading_error_rad": float(np.abs(col("heading")).mean()),
        "steering_sign_changes": int(np.sum(np.diff(np.sign(active)) != 0)),
        "mean_steering_rate_per_s": float(
            np.mean(np.abs(np.diff(steer)) / np.maximum(np.diff(t), 0.001))
        ),
        "average_speed_mps": float(col("speed").mean()),
        "mean_loop_ms": float(col("loop_ms").mean()),
        "p95_loop_ms": float(np.percentile(col("loop_ms"), 95)),
        "mean_control_ms": float(col("control_ms").mean()),
        "gate_rejects": sum(row["gate"] != "ok" for row in rows),
        "route_completion": "not measured: no ground-truth route/finish line",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", default="output/diagnostics/beamng_comparison.json")
    args = parser.parse_args()
    report = {"kind": __doc__, "results": [summarize(path) for path in args.logs]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
