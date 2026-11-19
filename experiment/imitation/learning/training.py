"""Reproducible supervised steering-proxy training, with validation-only checkpoint selection."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from experiment.imitation.config import ImitationConfig
from experiment.imitation.learning.dataset import DrivingDataset, read_manifest, steering_sampler
from experiment.imitation.learning.evaluation import predict_rows, steering_metrics
from experiment.imitation.learning.model import SteeringNetwork, resolve_device, save_checkpoint


def train(directory: Path, settings: ImitationConfig):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.set_num_threads(settings.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = read_manifest(directory)
    metadata = json.loads((directory / "dataset_metadata.json").read_text())
    for key in ("image_width", "image_height", "crop_top", "crop_bottom", "label_scale"):
        if metadata["settings"][key] != asdict(settings)[key]:
            raise ValueError(f"Training preprocessing differs from extracted dataset: {key}")
    training_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not training_rows or not validation_rows:
        raise ValueError("Training and validation splits must be nonempty")
    device = resolve_device(settings.device)
    dataset = DrivingDataset(directory, training_rows, settings, augment=True)
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        sampler=steering_sampler(training_rows, settings, generator),
        num_workers=settings.workers,
        generator=generator,
    )
    model = SteeringNetwork(settings, pretrained=settings.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    output = Path(settings.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    checkpoint = Path(settings.checkpoint)
    dataset_hash = hashlib.sha256((directory / "manifest.jsonl").read_bytes()).hexdigest()
    hardware = "CPU"
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "settings": asdict(settings),
                "manifest_sha256": dataset_hash,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": platform.python_version(),
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "hardware": hardware,
            },
            indent=2,
        )
        + "\n"
    )
    with SummaryWriter(str(output / "tensorboard")) as writer:
        for epoch in range(settings.epochs):
            model.train()
            frozen = epoch < settings.frozen_epochs
            for parameter in model.features.parameters():
                parameter.requires_grad_(not frozen)
            if frozen:
                model.features.eval()
            squared_error = 0.0
            samples = 0
            for image, speed, target in loader:
                image, speed, target = image.to(device), speed.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(image, speed)
                loss = torch.nn.functional.mse_loss(prediction, target)
                loss.backward()
                optimizer.step()
                squared_error += loss.item() * len(target)
                samples += len(target)
            target, prediction = predict_rows(model, directory, validation_rows, settings, device)
            validation = steering_metrics(
                target, prediction, validation_rows, settings.min_turn_rad
            )
            epoch_result = {
                "epoch": epoch + 1,
                "sampled_train_mse": squared_error / samples,
                "validation_mae_rad": validation["mae_rad"],
                "validation_rmse_rad": validation["rmse_rad"],
            }
            history.append(epoch_result)
            for key, value in epoch_result.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch + 1)
            if validation["mae_rad"] < best:
                best = validation["mae_rad"]
                save_checkpoint(
                    checkpoint,
                    model,
                    settings,
                    {
                        "label_semantics": {
                            "target_kind": "measured_steering_proxy",
                            "human_control_confirmed": False,
                            "units_status": "dynamics_supported_radians_hypothesis",
                            "publisher_conversion_verified": False,
                        },
                        "dataset_manifest_sha256": dataset_hash,
                        "camera_calibration": metadata["sessions"][0]["calibration"],
                        "best_validation_mae_rad": best,
                        "selected_epoch": epoch + 1,
                    },
                )
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                f"Epoch {epoch + 1}/{settings.epochs}: Train MSE={squared_error / samples:.6f} Validation MAE={validation['mae_rad']:.6f} Best={best:.6f}",
                flush=True,
            )
    _loss_plot(output, history)
    return {"checkpoint": str(checkpoint), "best_validation_mae_rad": best, "history": history}


def _loss_plot(output, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        epochs,
        [np.sqrt(row["sampled_train_mse"]) for row in history],
        label="Augmented Sampled Training RMSE",
    )
    ax.plot(
        epochs, [row["validation_rmse_rad"] for row in history], label="Unweighted Validation RMSE"
    )
    ax.set(
        xlabel="Epoch", ylabel="Error (Provisional Radians)", title="Training And Validation Error"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
