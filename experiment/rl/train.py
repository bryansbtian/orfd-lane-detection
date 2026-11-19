#!/usr/bin/env python3
"""Train and compare a residual steering policy without BeamNG or extra dependencies.

Uses reward-scaled, top-direction Augmented Random Search with fixed observation
scales: https://arxiv.org/abs/1803.07055. This is policy search from episode returns,
not imitation learning. The bicycle model is only an initial screening environment.

    python -m experiment.rl.train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# The S-bend simulation is a script, not a package module.
sys.path.insert(0, str(ROOT / "scripts"))

from sbend_sim import SimResult, simulate

from experiment.config import load_experiment_config
from experiment.rl.config import RLConfig
from experiment.rl.controller import FEATURES, ResidualPolicy, RLController
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.types import PipelineConfig


def ars_update(
    weights: np.ndarray,
    objective: Callable[[np.ndarray], float],
    settings: RLConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    deltas = rng.normal(size=(settings.directions, len(weights)))
    returns = np.empty((settings.directions, 2))
    for index, delta in enumerate(deltas):
        returns[index, 0] = objective(weights + settings.exploration_std * delta)
        returns[index, 1] = objective(weights - settings.exploration_std * delta)
    if not np.isfinite(returns).all():
        raise ValueError("RL episode returns must be finite")
    top = np.argsort(returns.max(axis=1))[-settings.top_directions :]
    scale = float(returns[top].std())
    if scale <= np.finfo(float).eps:
        return weights.copy()
    gradient = (returns[top, 0] - returns[top, 1]) @ deltas[top]
    return weights + settings.learning_rate * gradient / (len(top) * scale)


def scenarios(settings: RLConfig, rng: np.random.Generator, count: int) -> list[dict]:
    episodes = []
    for _ in range(count):
        episodes.append(
            {
                "radius": float(rng.uniform(settings.radius_min_m, settings.radius_max_m)),
                "fps": float(rng.uniform(settings.fps_min, settings.fps_max)),
                "start_yaw": float(
                    rng.uniform(-settings.start_yaw_max_rad, settings.start_yaw_max_rad)
                ),
                "start_lateral_m": float(
                    rng.uniform(
                        -settings.start_lateral_max_m,
                        settings.start_lateral_max_m,
                    )
                ),
                "delay_frames": int(rng.integers(1, settings.delay_max_frames + 1)),
                "mirror": bool(rng.integers(0, 2)),
                "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
            }
        )
    return episodes


def rollout(config: PipelineConfig, episode: dict, weights: np.ndarray | None) -> SimResult:
    policy = None
    if weights is not None:
        policy = ResidualPolicy(weights, config.rl)

    def factory(camera, clock):
        if policy is None:
            return StanleyController(config, camera=camera, clock=clock)
        return RLController(config, camera=camera, policy=policy, clock=clock)

    return simulate(config, **episode, terminate_off_trail=True, controller_factory=factory)


def reward(result: SimResult, settings: RLConfig) -> float:
    # Include the initial misalignment: the regression's warmup would hide early failures.
    error = float(np.mean([entry["err"] for entry in result.log]))
    edge_fraction = result.edge_frames / len(result.log)
    loss = (
        settings.weight_error * error
        + settings.weight_edge * edge_fraction
        + settings.weight_steer_change * result.steer_jerk
    )
    if not result.completed:
        loss += settings.failure_penalty
    return -loss


def metrics(result: SimResult) -> dict:
    return {
        "completed": bool(result.completed),
        "mean_abs_error_m": float(np.mean([entry["err"] for entry in result.log])),
        "p95_error_m": float(np.percentile([entry["err"] for entry in result.log], 95)),
        "max_error_m": result.max_lateral_error,
        "edge_frames": result.edge_frames,
        "frames": len(result.log),
        "steering_change_per_frame": result.steer_jerk,
        "mean_control_ms": float(np.mean([entry["control_ms"] for entry in result.log])),
    }


def compare(config: PipelineConfig, episodes: list[dict], weights: np.ndarray) -> list[dict]:
    rows = []
    for index, episode in enumerate(episodes):
        baseline = rollout(config, episode, None)
        learned = rollout(config, episode, weights)
        row = {
            "scenario": episode,
            "stanley": metrics(baseline),
            "rl": metrics(learned),
            "stanley_reward": reward(baseline, config.rl),
            "rl_reward": reward(learned, config.rl),
        }
        rows.append(row)
        print(
            f"Evaluation {index + 1}/{len(episodes)}: "
            f"Stanley={row['stanley_reward']:.4f} RL={row['rl_reward']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiment/configs/rl.yaml"))
    parser.add_argument("--out", help="Checkpoint path (default: control.rl.policy_path).")
    parser.add_argument("--report", default="output/rl/comparison.json")
    parser.add_argument("--iterations", type=int, help="Override control.rl.iterations.")
    parser.add_argument("--seed", type=int, help="Override control.rl.seed.")
    parser.add_argument("--evaluate", metavar="POLICY", help="Compare an existing policy only.")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    overrides = {}
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.seed is not None:
        overrides["seed"] = args.seed
    config.rl = replace(config.rl, **overrides)
    settings = config.rl
    # Independent streams keep evaluation identical across training budgets.
    train_seed, search_seed, eval_seed = np.random.SeedSequence(settings.seed).spawn(3)
    episodes = scenarios(settings, np.random.default_rng(train_seed), settings.train_episodes)
    evaluation = scenarios(settings, np.random.default_rng(eval_seed), settings.eval_episodes)
    history = []
    policy_path = Path(settings.policy_path)
    if args.out:
        policy_path = Path(args.out)
    if args.evaluate:
        policy_path = Path(args.evaluate)
        weights = ResidualPolicy.load(policy_path, settings).weights
    else:

        def objective(candidate):
            return float(
                np.mean(
                    [reward(rollout(config, episode, candidate), settings) for episode in episodes]
                )
            )

        rng = np.random.default_rng(search_seed)
        weights = np.zeros(len(FEATURES))
        best = weights.copy()
        best_reward = objective(best)
        initial_reward = best_reward
        for iteration in range(settings.iterations):
            weights = ars_update(weights, objective, settings, rng)
            score = objective(weights)
            if score > best_reward:
                best, best_reward = weights.copy(), score
            history.append(
                {"iteration": iteration + 1, "reward": score, "best_reward": best_reward}
            )
            print(
                f"Iteration {iteration + 1}/{settings.iterations}: "
                f"Reward={score:.4f} Best={best_reward:.4f}",
                flush=True,
            )
        weights = best
        ResidualPolicy(weights, settings).save(policy_path)
        print(
            f"Policy Saved: {policy_path} Training Reward: {initial_reward:.4f} -> {best_reward:.4f}"
        )
    rows = compare(config, evaluation, weights)
    baseline_reward = float(np.mean([row["stanley_reward"] for row in rows]))
    learned_reward = float(np.mean([row["rl_reward"] for row in rows]))
    report = {
        "config_path": str(Path(args.config).resolve()),
        "environment": "Synthetic S-Bend Bicycle Model",
        "limitation": "Offline screening only; no BeamNG or real-world improvement is established.",
        "settings": asdict(settings),
        "policy_path": str(policy_path),
        "weights": weights.tolist(),
        "training_scenarios": episodes,
        "training_history": history,
        "evaluation": rows,
        "stanley_mean_reward": baseline_reward,
        "rl_mean_reward": learned_reward,
        "rl_better_on_mean_reward": learned_reward > baseline_reward,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Comparison Saved: {report_path} Stanley={baseline_reward:.4f} RL={learned_reward:.4f}")


if __name__ == "__main__":
    main()
