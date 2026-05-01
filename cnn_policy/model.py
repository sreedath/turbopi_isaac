"""Tiny CNN policy for image-based path following."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from . import DEFAULT_FRAME_HISTORY, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH


@dataclass(frozen=True)
class LoopPolicyConfig:
    """Architecture and input-shape settings for the CNN policy."""

    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    frame_history: int = DEFAULT_FRAME_HISTORY
    action_dim: int = 3
    hidden_dim: int = 32
    dropout: float = 0.1
    # Channel widths for the 4 conv blocks. Defaults total ~60k params for
    # the (frame_history=3, image=64x48) configuration. Bump for more capacity.
    conv_channels: tuple[int, int, int, int] = (16, 32, 56, 64)
    head_dims: tuple[int, int] = (32, 16)

    @property
    def input_channels(self) -> int:
        return self.frame_history * 3


class ConvBlock(nn.Module):
    """Conv-BN-ReLU helper block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoopPolicyNet(nn.Module):
    """Tiny CNN that regresses normalized [vx, vy, omega]."""

    def __init__(self, config: LoopPolicyConfig | None = None):
        super().__init__()
        self.config = config or LoopPolicyConfig()
        c1, c2, c3, c4 = self.config.conv_channels
        self.encoder = nn.Sequential(
            ConvBlock(self.config.input_channels, c1, kernel_size=5, stride=2, padding=2),
            ConvBlock(c1, c2, kernel_size=3, stride=2, padding=1),
            ConvBlock(c2, c3, kernel_size=3, stride=2, padding=1),
            ConvBlock(c3, c4, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        h1, h2 = self.config.head_dims
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, self.config.action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, got {x.shape[1]}. "
                f"Frame history={self.config.frame_history}"
            )
        return self.head(self.encoder(x))


LoopCNNModel = LoopPolicyNet


def build_model(config: LoopPolicyConfig | None = None) -> LoopPolicyNet:
    """Create a fresh CNN policy model."""
    return LoopPolicyNet(config=config)


def save_checkpoint(
    path: Path,
    model: LoopPolicyNet,
    *,
    epoch: int,
    metrics: dict[str, float],
    extra: dict[str, object] | None = None,
) -> None:
    """Persist a checkpoint with model and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device | None = None) -> tuple[LoopPolicyNet, dict[str, object]]:
    """Load a checkpoint and return the instantiated model plus raw payload."""
    payload = torch.load(Path(path), map_location=map_location)
    config = LoopPolicyConfig(**payload.get("model_config", {}))
    model = LoopPolicyNet(config=config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload

