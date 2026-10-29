#!/usr/bin/env python3
"""Archive Exact Frozen Sources, Accepting Historical Recovery Only By SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/r3/tracking_improvement/baseline_source"


def variants(path, existing):
    for source in (existing, ROOT / path, BASE / path):
        if source.exists():
            raw = source.read_bytes()
            yield raw
            text = raw.decode("utf-8").replace("\r\n", "\n")
            # These deletions recover earlier development versions. A result is
            # retained only if its complete bytes match the recorded run hash.
            text = text.replace(
                "        self._censored_weight = config.baseline_censored_row_weight\n", ""
            )
            text = text.replace("        weights: list[float] = []\n", "")
            text = text.replace(
                "            weight = 1.0\n            if run[0] <= self._edge_margin_px or run[1] >= w - 1 - self._edge_margin_px:\n                weight = self._censored_weight\n            weights.append(weight)\n",
                "",
            )
            text = text.replace(
                "        # Weight squared residuals by observation reliability. A clipped run\n        # still establishes continuity, but does not establish its true midpoint.\n        coeffs = np.polyfit(forward, right, deg=degree, w=np.sqrt(weights))\n",
                "        coeffs = np.polyfit(forward, right, deg=degree)\n",
            )
            text = text.replace("    baseline_censored_row_weight: float = 1.0\n", "")
            text = text.replace(
                '        baseline_censored_row_weight=_positive(plan, "baseline_censored_row_weight", 1.0),\n',
                "",
            )
            text = text.replace(
                "  # Keep 1.0 for the original fit. Development overlays can reduce a cropped\n  # row's influence without removing its evidence of a drivable continuation.\n  baseline_censored_row_weight: 1.0\n",
                "",
            )
            text = text.replace(
                '            result["camera_world_position"] = self._cameras["dashcam"].get_position()\n',
                "",
            )
            text = text.replace(
                '            result["camera_world_direction"] = self._cameras["dashcam"].get_direction()\n',
                "",
            )
            if path == "scripts/r3_verify.py":
                begin = text.find(
                    '    files.extend((ROOT / "src/offroad_autonomy/runtime").glob("r3_*.py"))'
                )
                if begin >= 0:
                    end = text.index("    return {", begin)
                    text = text[:begin] + text[end:]
            yield text.encode("utf-8")
            yield text.replace("\n", "\r\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads((args.batch / "experiment.json").read_text())
    directory = args.batch / "source_snapshot"
    directory.mkdir(exist_ok=True)
    missing = []
    files = {}
    for name, expected in experiment["source_hashes"].items():
        path = name.replace("\\", "/")
        if path.endswith(".pt"):
            files[path] = {
                "sha256": expected,
                "storage": "Original Model File; Hash Only In This Archive",
            }
            continue
        found = False
        for raw in variants(path, directory / path):
            if hashlib.sha256(raw).hexdigest() == expected:
                target = directory / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                files[path] = {"sha256": expected, "storage": "Exact Archived Bytes"}
                found = True
                break
        if not found:
            missing.append(path)
    (directory / "manifest.json").write_text(
        json.dumps({"files": files, "unrecovered": missing}, indent=2)
    )
    print(json.dumps({"batch": str(args.batch), "archived": len(files), "unrecovered": missing}))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
