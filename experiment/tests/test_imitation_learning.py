"""Optional learning dependency tests exercise real checkpoints and deliberate leakage failures."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torchvision")
pytest.importorskip("tensorboard")

from experiment.imitation.config import ImitationConfig
from experiment.imitation.learning.dataset import DrivingDataset, read_manifest
from experiment.imitation.learning.evaluation import (
    evaluate_checkpoint,
    persistence_predictions,
    steering_metrics,
)
from experiment.imitation.learning.model import SteeringPredictor
from experiment.imitation.learning.training import train


@pytest.fixture
def dataset(tmp_path):
    settings = replace(
        ImitationConfig(),
        image_width=64,
        image_height=32,
        epochs=1,
        frozen_epochs=1,
        batch_size=4,
        pretrained=False,
        device="cpu",
        cpu_threads=1,
        checkpoint=str(tmp_path / "model.pt"),
        run_dir=str(tmp_path / "run"),
        latency_warmup=1,
        latency_samples=2,
    )
    rows = []
    for index in range(16):
        split = "train"
        if index >= 8:
            split = "validation"
        if index >= 12:
            split = "test"
        image = np.full((32, 64, 3), index * 10, dtype=np.uint8)
        image[:, : index + 1, 0] = 255
        frame = f"frame{index}.jpg"
        cv2.imwrite(str(tmp_path / frame), image)
        timestamp = 1_000_000_000 + index * 100_000_000
        rows.append(
            {
                "session_id": "fixture",
                "split": split,
                "image_header_ns": timestamp,
                "target_header_ns": timestamp,
                "speed_header_ns": timestamp - 1,
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "frame": frame,
                "steering_proxy_rad": (index - 8) * 0.01,
                "speed_mps": 2.0,
                "target_kind": "measured_steering_proxy",
            }
        )
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "settings": asdict(settings),
                "sessions": [{"calibration": {"width": 2880, "k": [1714.20536]}}],
            }
        )
    )
    return tmp_path, rows, settings


def test_manifest_refuses_duplicate_pixels_future_state_and_path_escape(dataset):
    directory, rows, _ = dataset
    assert len(read_manifest(directory)) == 16
    rows[1]["pixel_sha256"] = rows[0]["pixel_sha256"]
    (directory / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(directory)
    rows[1]["pixel_sha256"] = "unique"
    rows[1]["speed_header_ns"] = rows[1]["image_header_ns"] + 1
    (directory / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Future"):
        read_manifest(directory)
    rows[1]["speed_header_ns"] = rows[1]["image_header_ns"]
    rows[1]["frame"] = "../outside.jpg"
    (directory / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="escapes"):
        read_manifest(directory)


def test_flip_augmentation_changes_steering_sign_consistently(dataset):
    directory, rows, settings = dataset
    settings = replace(settings, flip_probability=1, color_jitter=0)
    original = DrivingDataset(directory, rows, settings)
    augmented = DrivingDataset(directory, rows, settings, augment=True)
    image, _, angle = original[0]
    mirrored, _, opposite = augmented[0]
    assert angle == -opposite
    np.testing.assert_allclose(image.flip(-1).numpy(), mirrored.numpy())


def test_real_training_checkpoint_cpu_inference_and_evaluation_roundtrip(dataset):
    directory, _, settings = dataset
    result = train(directory, settings)
    assert Path(result["checkpoint"]).is_file()
    predictor = SteeringPredictor(settings.checkpoint, "cpu")
    assert not predictor.metadata["label_semantics"]["human_control_confirmed"]
    value = predictor.predict(np.zeros((62, 96, 3), dtype=np.uint8), 2)
    assert np.isfinite(value)
    report = evaluate_checkpoint(
        Path(settings.checkpoint), directory, directory / "evaluation", device="cpu"
    )
    assert report["imitation"]["samples"] == 4
    assert report["network_latency_ms"]["mean"] > 0
    assert (directory / "evaluation" / "steering_comparison.png").is_file()


def test_persistence_never_reuses_an_unavailable_future_label(dataset):
    _, rows, _ = dataset
    rows = rows[:3]
    rows[0]["target_header_ns"] = rows[1]["image_header_ns"] + 1
    predicted = persistence_predictions(rows)
    assert predicted[0] == 0
    assert predicted[1] == 0
    assert predicted[2] == rows[1]["steering_proxy_rad"]


def test_metrics_separate_turns_and_smoothness(dataset):
    _, rows, _ = dataset
    target = np.array([0, 0.1, -0.1])
    result = steering_metrics(target, np.zeros(3), rows[:3], 0.03)
    assert result["mae_rad"] == pytest.approx(0.2 / 3)
    assert result["turn_mae_rad"] == pytest.approx(0.1)
    assert result["straight_mae_rad"] == 0
    assert result["mean_abs_steering_rate_rad_s"] == 0
