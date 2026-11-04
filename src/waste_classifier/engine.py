"""Shared train and evaluation loop."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    """Run one pass over `loader`; trains if `optimizer` is given.

    Returns mean loss, true labels and predicted labels.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, y_true, y_pred = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            y_true += labels.tolist()
            y_pred += logits.argmax(dim=1).tolist()

    return total_loss / len(y_true), y_true, y_pred
