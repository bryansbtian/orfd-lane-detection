"""MobileNetV3 steering regression, shared preprocessing, and portable checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from experiment.imitation.config import ImitationConfig
from experiment.imitation.learning.extraction import crop_resize

MODEL_VERSION = 1
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_tensor(processed_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(processed_bgr[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cpu"
        if torch.cuda.is_available():
            value = "cuda"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(value)


class SteeringNetwork(nn.Module):
    def __init__(self, settings: ImitationConfig, pretrained: bool = False):
        super().__init__()
        weights = None
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(backbone.classifier[0].in_features + 1, 128),
            nn.Hardswish(),
            nn.Dropout(settings.dropout),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        # Begin at the zero baseline so pretrained visual features cannot cause large initial turns.
        nn.init.zeros_(self.head[3].weight)
        nn.init.zeros_(self.head[3].bias)
        self.max_angle = settings.max_angle_rad
        self.speed_scale = settings.speed_scale_mps

    def forward(self, image: torch.Tensor, speed: torch.Tensor):
        features = self.pool(self.features(image)).flatten(1)
        speed = speed.reshape(-1, 1) / self.speed_scale
        return self.head(torch.cat([features, speed], dim=1)).squeeze(1) * self.max_angle


class SteeringPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.device = resolve_device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            data.get("version") != MODEL_VERSION
            or data.get("model_name") != "mobilenet_v3_small_speed"
        ):
            raise ValueError("Unsupported imitation checkpoint")
        if data.get("label_semantics", {}).get("target_kind") != "measured_steering_proxy":
            raise ValueError("Checkpoint does not document steering label semantics")
        self.settings = ImitationConfig(**data["settings"])
        torch.set_num_threads(self.settings.cpu_threads)
        self.model = SteeringNetwork(self.settings).to(self.device)
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.eval()
        self.metadata = {key: value for key, value in data.items() if key != "state_dict"}
        with torch.inference_mode():
            image = torch.zeros(
                (1, 3, self.settings.image_height, self.settings.image_width), device=self.device
            )
            speed = torch.zeros(1, device=self.device)
            for _ in range(self.settings.latency_warmup):
                self.model(image, speed)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    @torch.inference_mode()
    def predict(self, raw_bgr: np.ndarray, speed_mps: float) -> float:
        image = (
            normalize(image_tensor(crop_resize(raw_bgr, self.settings)))
            .unsqueeze(0)
            .to(self.device)
        )
        speed = torch.tensor([speed_mps], dtype=torch.float32, device=self.device)
        value = self.model(image, speed)
        return float(value.cpu().item())


def save_checkpoint(path: Path, model: SteeringNetwork, settings: ImitationConfig, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "version": MODEL_VERSION,
            "model_name": "mobilenet_v3_small_speed",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "settings": asdict(settings),
            **metadata,
        },
        temporary,
    )
    temporary.replace(path)
