"""Lazy image loading with explicit split validation and steering-aware sampling."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import ColorJitter

from experiment.imitation.config import ImitationConfig
from experiment.imitation.learning.model import image_tensor, normalize


def read_manifest(directory: Path) -> list[dict]:
    rows = []
    split_hashes = {}
    split_times = {}
    with (directory / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            # Older Windows manifests must remain readable when a dataset moves to a Jetson.
            row["frame"] = row["frame"].replace("\\", "/")
            split = row["split"]
            if split not in {"train", "validation", "test"}:
                raise ValueError("Dataset has an invalid split")
            key = (row["session_id"], row["image_header_ns"])
            digest = row["pixel_sha256"]
            if key in split_times or digest in split_hashes:
                raise ValueError("Dataset contains duplicate frames or cross-split leakage")
            split_times[key] = split
            split_hashes[digest] = split
            if row["speed_header_ns"] > row["image_header_ns"]:
                raise ValueError("Future vehicle state leaked into an input")
            resolved = (directory / row["frame"]).resolve()
            if not resolved.is_relative_to(directory.resolve()):
                raise ValueError("Dataset frame escapes its directory")
            if row["target_kind"] != "measured_steering_proxy":
                raise ValueError("Dataset steering semantics are unsupported")
            if not all(
                math.isfinite(float(row[key])) for key in ("speed_mps", "steering_proxy_rad")
            ):
                raise ValueError("Dataset inputs and targets must be finite")
            rows.append(row)
    if not rows:
        raise ValueError("Dataset manifest is empty")
    return rows


class DrivingDataset(Dataset):
    def __init__(self, directory: Path, rows: list[dict], settings: ImitationConfig, augment=False):
        self.directory = directory
        self.rows = rows
        self.settings = settings
        self.augment = augment
        self.jitter = ColorJitter(
            settings.color_jitter, settings.color_jitter, settings.color_jitter
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.directory / row["frame"]))
        if image is None:
            raise OSError(f"Cannot read dataset frame: {row['frame']}")
        if image.shape[:2] != (self.settings.image_height, self.settings.image_width):
            raise ValueError("Dataset image size does not match checkpoint preprocessing")
        image = image_tensor(image)
        target = float(row["steering_proxy_rad"])
        if self.augment:
            image = self.jitter(image)
            if random.random() < self.settings.flip_probability:
                image = image.flip(-1)
                target = -target
        return (
            normalize(image),
            torch.tensor(row["speed_mps"], dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def steering_sampler(rows: list[dict], settings: ImitationConfig, generator: torch.Generator):
    weights = []
    for row in rows:
        weight = 1.0
        if abs(row["steering_proxy_rad"]) >= settings.min_turn_rad:
            weight = settings.turn_sampling_weight
        weights.append(weight)
    return WeightedRandomSampler(weights, len(rows), replacement=True, generator=generator)
