"""Model creation and checkpoint I/O."""

from pathlib import Path

import timm
import torch
from torch import nn


class BasicCNN(nn.Module):
    """Model A: compact CNN trained from scratch.

    It outputs logits. Softmax is applied by the loss during training and at prediction time.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]
        for c_in, c_out in zip(channels, channels[1:], strict=False):
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "basic_cnn":
        return BasicCNN(num_classes)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except a timm model's classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": cfg["model"]["name"],
            "classes": cfg["data"]["classes"],
            "image_size": cfg["data"]["image_size"],
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Return the model in eval mode, plus the checkpoint's metadata."""
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint.pop("state_dict"))
    return model.to(device).eval(), checkpoint
