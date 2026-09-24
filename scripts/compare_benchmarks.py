#!/usr/bin/env python3
"""Tabulate benchmark reports from several runs side by side.

    python scripts/compare_benchmarks.py output/benchmarks/*.json

Each report comes from one ``--benchmark-seconds`` run of
``offroad-autonomy``. Values the run could
not measure - stereo metrics with stereo off, say - print as "-".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = (
    ("run", lambda r: r.get("label")),
    ("FPS", lambda r: r.get("main_fps")),
    ("lat mean", lambda r: r.get("primary_latency_mean_ms")),
    ("lat p95", lambda r: r.get("primary_latency_p95_ms")),
    ("loop p95", lambda r: r.get("full_loop_p95_ms")),
    ("stereo Hz", lambda r: r.get("stereo", {}).get("fps")),
    ("stereo ms", lambda r: r.get("stereo", {}).get("latency_mean_ms")),
    ("seg conf", lambda r: r.get("segmentation_confidence_mean")),
    ("disp %", lambda r: r.get("valid_disparity_pct_mean")),
    ("cover %", lambda r: r.get("depth_coverage_pct_mean")),
    ("depth used %", lambda r: r.get("depth_used_pct")),
    ("jitter", lambda r: r.get("path_jitter_pct")),
    ("departures", lambda r: r.get("lane_departures")),
    (">=20FPS", lambda r: _yes_no(r.get("meets_20fps"))),
    ("<=50ms", lambda r: _yes_no(r.get("meets_50ms"))),
)


def _yes_no(flag) -> str:
    if flag:
        return "yes"
    return "no"


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 10:
            return f"{value:.2f}"
        return f"{value:.1f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.reports):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append([_cell(fn(report)) for _, fn in COLUMNS])

    headers = [name for name, _ in COLUMNS]
    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
