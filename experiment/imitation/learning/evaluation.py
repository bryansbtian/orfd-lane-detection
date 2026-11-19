"""Unweighted, chronologically ordered open-loop evaluation with honest proxy baselines."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment.imitation.learning.dataset import DrivingDataset


def steering_metrics(target, prediction, rows, turn_threshold):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - target
    result = {
        "samples": len(error),
        "mae_rad": float(abs(error).mean()),
        "rmse_rad": float(np.sqrt(np.square(error).mean())),
    }
    for name, selection in (
        ("turn", abs(target) >= turn_threshold),
        ("straight", abs(target) < turn_threshold),
    ):
        result[name + "_samples"] = int(selection.sum())
        result[name + "_mae_rad"] = None
        if selection.any():
            result[name + "_mae_rad"] = float(abs(error[selection]).mean())
    rates = []
    for index in range(1, len(rows)):
        if rows[index]["session_id"] != rows[index - 1]["session_id"]:
            continue
        dt = (rows[index]["image_header_ns"] - rows[index - 1]["image_header_ns"]) / 1e9
        if dt > 0:
            rates.append(abs(prediction[index] - prediction[index - 1]) / dt)
    result["mean_abs_steering_rate_rad_s"] = None
    if rates:
        result["mean_abs_steering_rate_rad_s"] = float(np.mean(rates))
    return result


@torch.inference_mode()
def predict_rows(model, directory, rows, settings, device):
    dataset = DrivingDataset(directory, rows, settings)
    loader = DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.workers
    )
    predictions = []
    targets = []
    model.eval()
    for image, speed, target in loader:
        predictions.extend(model(image.to(device), speed.to(device)).cpu().tolist())
        targets.extend(target.tolist())
    return np.asarray(targets), np.asarray(predictions)


def persistence_predictions(rows):
    predictions = []
    prior = None
    for row in rows:
        prediction = 0.0
        # A previous measured target must already exist at the current observation time.
        if (
            prior is not None
            and prior["session_id"] == row["session_id"]
            and prior["target_header_ns"] <= row["image_header_ns"]
        ):
            prediction = prior["steering_proxy_rad"]
        predictions.append(prediction)
        prior = row
    return np.asarray(predictions)


def evaluate_checkpoint(
    checkpoint: Path, directory: Path, output: Path, split="test", device="auto"
):
    from experiment.imitation.learning.dataset import read_manifest
    from experiment.imitation.learning.model import SteeringPredictor

    output.mkdir(parents=True, exist_ok=True)
    predictor = SteeringPredictor(checkpoint, device)
    rows = [row for row in read_manifest(directory) if row["split"] == split]
    if not rows:
        raise ValueError(f"Dataset has no {split} samples")
    targets, predictions = predict_rows(
        predictor.model, directory, rows, predictor.settings, predictor.device
    )
    persistence = persistence_predictions(rows)
    report = {
        "split": split,
        "target_kind": "measured_steering_proxy",
        "human_control_confirmed": False,
        "imitation": steering_metrics(targets, predictions, rows, predictor.settings.min_turn_rad),
        "zero": steering_metrics(
            targets, np.zeros_like(targets), rows, predictor.settings.min_turn_rad
        ),
        "persistence": steering_metrics(
            targets, persistence, rows, predictor.settings.min_turn_rad
        ),
        "stanley": {
            "status": "Not Comparable",
            "reason": "Real camera extrinsics and a calibrated controller-to-recording angle mapping are unavailable",
        },
        "rl": {
            "status": "Not Comparable",
            "reason": "Path-based residual policy has no calibrated real-camera path/actuation ground truth",
        },
        "device": str(predictor.device),
    }
    sample = DrivingDataset(directory, rows[:1], predictor.settings)[0]
    image, speed = (
        sample[0].unsqueeze(0).to(predictor.device),
        sample[1].reshape(1).to(predictor.device),
    )
    measurements = []
    with torch.inference_mode():
        for index in range(predictor.settings.latency_warmup + predictor.settings.latency_samples):
            if predictor.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            predictor.model(image, speed)
            if predictor.device.type == "cuda":
                torch.cuda.synchronize()
            if index >= predictor.settings.latency_warmup:
                measurements.append((time.perf_counter() - started) * 1000)
    report["network_latency_ms"] = {
        "mean": float(np.mean(measurements)),
        "p95": float(np.percentile(measurements, 95)),
        "includes_preprocessing": False,
    }
    camera = predictor.metadata["camera_calibration"]
    native = np.zeros(
        (camera.get("height", predictor.settings.image_height), camera["width"], 3), dtype=np.uint8
    )
    full_measurements = []
    for _ in range(predictor.settings.latency_samples):
        started = time.perf_counter()
        predictor.predict(native, rows[0]["speed_mps"])
        full_measurements.append((time.perf_counter() - started) * 1000)
    report["preprocessing_and_inference_latency_ms"] = {
        "mean": float(np.mean(full_measurements)),
        "p95": float(np.percentile(full_measurements, 95)),
        "input": "Synthetic Native-Resolution BGR Image; Includes Crop, Resize, Transfer, And Prediction",
        "includes_video_decode": False,
    }
    report["time_bins"] = []
    for indices in np.array_split(np.arange(len(rows)), min(10, len(rows))):
        report["time_bins"].append(
            steering_metrics(
                targets[indices],
                predictions[indices],
                [rows[index] for index in indices],
                predictor.settings.min_turn_rad,
            )
        )
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["session", "image_header_ns", "target_proxy_rad", "prediction_rad", "persistence_rad"]
        )
        for row, target, prediction, prior in zip(rows, targets, predictions, persistence):
            writer.writerow([row["session_id"], row["image_header_ns"], target, prediction, prior])
    _plots(output, rows, targets, predictions, persistence, report)
    return report


def _plots(output, rows, targets, predictions, persistence, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([row["image_header_ns"] for row in rows], dtype=np.int64)
    times = (times - times[0]) / 1e9
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(times, targets, label="Measured Steering Proxy", linewidth=1)
    axes[0].plot(times, predictions, label="Imitation Prediction", linewidth=1)
    axes[0].set(
        xlabel="Elapsed Time (Seconds)",
        ylabel="Steering (Provisional Radians)",
        title="Temporally Held-Out Steering Proxy",
    )
    axes[0].legend()
    names = ["zero", "persistence", "imitation"]
    axes[1].bar(
        ["Constant Zero", "Previous Steering", "Imitation"],
        [report[name]["mae_rad"] for name in names],
    )
    axes[1].set(ylabel="MAE (Provisional Radians)", title="Open-Loop Proxy Comparison")
    fig.tight_layout()
    fig.savefig(output / "steering_comparison.png", dpi=150)
    plt.close(fig)
