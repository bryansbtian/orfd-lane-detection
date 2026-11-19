"""Drive BeamNG with an experimental planner or controller.

    python -m experiment.run --config experiment/configs/grid.yaml
    python -m experiment.run --controller rl

Every flag of ``offroad-autonomy`` works here too. The loop, safe stop and
shutdown are the default stack's; only the planner and controller change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from experiment.config import CONTROLLERS, PLANNERS, ExperimentConfig, load_experiment_config
from experiment.pipeline import ExperimentPipeline
from offroad_autonomy import main as default_main
from offroad_autonomy.utils.logger import setup_logger

logger = logging.getLogger("experiment.run")


def build_parser() -> argparse.ArgumentParser:
    parser = default_main.build_parser()
    parser.description = "Experimental planners and controllers on the offroad_autonomy stack"
    parser.add_argument("--planner", choices=PLANNERS, help="Override experiment.planner.")
    parser.add_argument("--controller", choices=CONTROLLERS, help="Override experiment.controller.")
    parser.add_argument("--rl-policy", help="RL checkpoint path; selects the RL controller.")
    parser.add_argument(
        "--imitation-model", help="Imitation checkpoint path; selects the imitation controller."
    )
    parser.add_argument(
        "--imitation-calibration",
        help="Measured simulation steering calibration JSON; selects the imitation controller.",
    )
    parser.add_argument(
        "--imitation-experiment",
        action="store_true",
        help="Allow a measured simulator adapter with explicitly unverified recording conventions.",
    )
    return parser


def _select_controller(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.planner:
        config.planner_mode = args.planner
    if args.controller:
        config.control_mode = args.controller
    imitation_flags = (
        args.imitation_model or args.imitation_calibration or args.imitation_experiment
    )
    if imitation_flags:
        if args.controller and args.controller != "imitation":
            sys.exit("Imitation flags conflict with the selected controller")
        config.control_mode = "imitation"
        if args.imitation_experiment:
            config.imitation = replace(config.imitation, allow_simulation_experiment=True)
        if args.imitation_model:
            config.imitation = replace(config.imitation, checkpoint=args.imitation_model)
        if args.imitation_calibration:
            config.imitation = replace(
                config.imitation, calibration_path=args.imitation_calibration
            )
    if args.rl_policy:
        if (args.controller and args.controller != "rl") or imitation_flags:
            sys.exit("RL flags conflict with the selected controller")
        config.control_mode = "rl"
        config.rl = replace(config.rl, policy_path=args.rl_policy)


def _check_imitation(config: ExperimentConfig) -> None:
    from experiment.imitation.controller import SteeringCalibration

    try:
        calibration = SteeringCalibration.load(
            config.imitation.calibration_path,
            allow_simulation_experiment=config.imitation.allow_simulation_experiment,
        )
        calibration.validate_simulator(config)
        if calibration.experimental:
            config.wheelbase_m = calibration.simulator_wheelbase
            logger.warning(
                "Simulation Experiment: Measured Simulator Adapter; "
                "Recording Units, Direction, Wheelbase And Camera Extrinsics Remain Unverified"
            )
        if not Path(config.imitation.checkpoint).is_file():
            raise ValueError(
                "Imitation checkpoint not found; run python -m experiment.imitation train first"
            )
    except ValueError as exc:
        sys.exit(str(exc))
    logger.warning(
        "Experimental Measured-Steering Proxy Enabled; Human Control Provenance Is Unverified"
    )


def _check_rl(config: ExperimentConfig) -> None:
    from experiment.rl.controller import ResidualPolicy

    try:
        ResidualPolicy.load(config.rl.policy_path, config.rl)
    except ValueError as exc:
        sys.exit(str(exc))
    logger.warning("Experimental RL Steering Enabled: %s", config.rl.policy_path)


def run_label(config: ExperimentConfig) -> str:
    """Presentation videos must not pass an experiment off as the default stack."""
    if config.control_mode == "imitation":
        if config.imitation.allow_simulation_experiment:
            return "Imitation Learning | Simulation Experiment"
        return "Imitation Learning"
    return f"Experiment | {config.planner_mode} + {config.control_mode}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logger(level=getattr(logging, args.log_level))

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Configuration file not found: {config_path}")
    try:
        config = load_experiment_config(config_path)
    except ValueError as exc:
        sys.exit(str(exc))
    _select_controller(config, args)
    # Validate before loading perception or opening a simulator connection.
    if config.control_mode == "imitation":
        _check_imitation(config)
    if config.control_mode == "rl":
        _check_rl(config)
    default_main.apply_overrides(config, args)
    logger.info("Experiment: planner %s, controller %s", config.planner_mode, config.control_mode)
    default_main.run(args, config, build_pipeline=ExperimentPipeline, run_label=run_label(config))


if __name__ == "__main__":
    main()
