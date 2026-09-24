#!/usr/bin/env python3
"""Export the segmentation model to a TensorRT engine with the prompts baked in.

TensorRT engines are specific to the GPU and TensorRT version that built
them, so run this on the Jetson itself (inside the container)::

    python scripts/export_engine.py --weights models/yoloe-26s-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from offroad_autonomy.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config whose prompts and input size are baked in.",
    )
    parser.add_argument("--weights", default="models/yoloe-26s-seg.pt")
    parser.add_argument(
        "--fp32", action="store_true", help="Build a full-precision engine instead of FP16."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = YOLO(args.weights)
    # An engine cannot be re-prompted at runtime, so the classes must be set
    # before export or the engine segments the checkpoint's default classes.
    model.set_classes(list(config.perception_prompts))
    engine = model.export(
        format="engine",
        imgsz=config.perception_input_size,
        half=not args.fp32,
        device=0,
    )
    print(f"Engine written to {Path(engine)}")


if __name__ == "__main__":
    main()
