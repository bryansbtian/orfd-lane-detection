"""Interleave saved-baseline and current postprocessing on identical RGB labels.

This isolates CPU postprocessing cost from GPU clocks and inference noise.
Annotations supply realistic mask shapes here, never perception accuracy claims.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import PerceptionResult, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.baseline_root / "src/offroad_autonomy/postprocessing/temporal_stabilizer.py"
    spec = importlib.util.spec_from_file_location("baseline_stabilizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = PipelineConfig(enable_ema=False, enable_morphology=False)
    before, after = module.TemporalStabilizer(config), TemporalStabilizer(config)
    results = []
    for path in sorted(args.labels.glob("*.png"))[:30]:
        label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(label, (720, 465), interpolation=cv2.INTER_NEAREST) == 255
        results.append(PerceptionResult(mask=mask))
    if not results:
        raise ValueError("No annotation-shaped masks found")
    samples = {"before": [], "after": []}
    for index in range(330):
        result = results[index % len(results)]
        pairs = [("before", before), ("after", after)]
        if index % 2:
            pairs.reverse()
        outputs = {}
        for name, stage in pairs:
            start = time.perf_counter()
            outputs[name] = stage.stabilize(result)
            elapsed = (time.perf_counter() - start) * 1000
            if index >= 30:
                samples[name].append(elapsed)
        np.testing.assert_array_equal(outputs["before"].mask, outputs["after"].mask)
        assert outputs["before"].stability_score == outputs["after"].stability_score
    report = {
        "kind": "interleaved_cpu_postprocessing_microbenchmark",
        "outputs_identical": True,
        "samples_per_variant": 300,
        "before_ms": {
            "mean": float(np.mean(samples["before"])),
            "p95": float(np.percentile(samples["before"], 95)),
        },
        "after_ms": {
            "mean": float(np.mean(samples["after"])),
            "p95": float(np.percentile(samples["after"], 95)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
