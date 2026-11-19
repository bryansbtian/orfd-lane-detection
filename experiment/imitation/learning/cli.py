"""Command line entry points for the offline imitation pipeline:
``python -m experiment.imitation <command>``."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiment.config import load_experiment_config


def main():
    parser = argparse.ArgumentParser(description="Off-Road Steering Proxy Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "train", "evaluate", "compare", "calibrate"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="experiment/configs/imitation.yaml")
        child.add_argument("--dataset")
        child.add_argument("--device", choices=["auto", "cpu", "cuda"])
        if name == "extract":
            child.add_argument("--archives", nargs="+")
        if name == "train":
            child.add_argument("--epochs", type=int)
        if name == "evaluate":
            child.add_argument("--split", choices=["validation", "test"], default="test")
        if name == "compare":
            child.add_argument("--suite", default="experiment/configs/imitation_comparison.yaml")
        if name == "calibrate":
            child.add_argument("--output", default="output/imitation/live_calibration")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    settings = config.imitation
    if args.device:
        settings = replace(settings, device=args.device)
    directory = Path(settings.dataset_dir)
    if args.dataset:
        directory = Path(args.dataset)
    if args.command == "extract":
        from experiment.imitation.learning.extraction import build_dataset

        archives = sorted(Path.cwd().glob("bag*.zip"))
        if args.archives:
            archives = [Path(value) for value in args.archives]
        result = build_dataset(archives, directory, settings)
        print(json.dumps({"counts": result["counts"], "rejected": result["rejected"]}, indent=2))
    elif args.command == "train":
        import torch

        from experiment.imitation.learning.training import train

        if args.epochs is not None:
            settings = replace(settings, epochs=args.epochs)
        # Keep downloaded weights with local models, outside source history.
        torch.hub.set_dir(str(Path(settings.checkpoint).parent / "torch_cache"))
        result = train(directory, settings)
        print(
            json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2)
        )
    elif args.command == "evaluate":
        from experiment.imitation.learning.evaluation import evaluate_checkpoint

        result = evaluate_checkpoint(
            Path(settings.checkpoint),
            directory,
            Path(settings.run_dir) / args.split,
            args.split,
            settings.device,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "calibrate":
        from experiment.imitation.calibration import measure_calibration
        from offroad_autonomy.utils.logger import setup_logger

        setup_logger()
        result = measure_calibration(config, Path(args.output))
        print(json.dumps(result, indent=2))
    else:
        from experiment.imitation.comparison import compare_suite

        result = compare_suite(config, Path(args.suite))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
