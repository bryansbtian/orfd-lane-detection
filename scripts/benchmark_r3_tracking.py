#!/usr/bin/env python3
"""Check R3 Candidates On Independent Synthetic Curves, Rates, And Noise Seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sbend_sim import load_config, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.configs:
        cfg = load_config(path)
        for radius in (9.0, 15.0, 25.0):
            for fps in (3.0, 5.0):
                for seed in (0, 1):
                    result = simulate(cfg, radius=radius, fps=fps, seed=seed)
                    row = {
                        "config": path,
                        "radius_m": radius,
                        "fps": fps,
                        "seed": seed,
                        "completed": result.completed,
                        "mean_absolute_error_m": result.mean_abs_error,
                        "maximum_error_m": result.max_lateral_error,
                        "edge_frames": result.edge_frames,
                        "steering_jerk_per_frame": result.steer_jerk,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"scope": "Synthetic; Not BeamNG Evidence", "scenarios": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
