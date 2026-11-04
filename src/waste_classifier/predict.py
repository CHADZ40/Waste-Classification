"""Classify a single image with a saved checkpoint."""

import argparse

import torch
from PIL import Image
from torch import nn

from waste_classifier.data import build_transforms
from waste_classifier.model import load_checkpoint
from waste_classifier.utils import get_device


@torch.inference_mode()
def predict_image(
    model: nn.Module, meta: dict, image: Image.Image, device: torch.device
) -> dict[str, float]:
    """Return each class with its confidence, highest first."""
    transform = build_transforms(meta["image_size"], train=False)
    batch = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = model(batch).softmax(dim=1)[0].tolist()
    scores = zip(meta["classes"], probabilities, strict=True)
    return dict(sorted(scores, key=lambda x: x[1], reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="models/transfer.pt")
    args = parser.parse_args()

    device = get_device()
    model, meta = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        scores = predict_image(model, meta, image, device)
    for label, score in scores.items():
        print(f"{label:<15} {score:.2%}")


if __name__ == "__main__":
    main()
