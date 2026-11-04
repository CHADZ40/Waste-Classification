"""Train one experiment and keep the checkpoint with the best val macro F1."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from waste_classifier.data import build_dataloader
from waste_classifier.engine import run_epoch
from waste_classifier.model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)
from waste_classifier.utils import checkpoint_path, get_device, load_config, seed_everything


def balanced_class_weights(labels: pd.Series, classes: list[str]) -> torch.Tensor:
    """Weight each class inversely to its frequency in the training labels."""
    counts = labels.value_counts().reindex(classes)
    if counts.isna().any():
        raise ValueError(f"Training data has no images for: {list(counts[counts.isna()].index)}")
    return torch.tensor((len(labels) / (len(classes) * counts)).values, dtype=torch.float32)


def train(cfg: dict) -> float:
    seed_everything(cfg["seed"])
    device = get_device()
    print(f"Experiment: {cfg['experiment']} | device: {device}")

    domains = cfg["data"]["train_domains"]
    classes = cfg["data"]["classes"]
    train_loader = build_dataloader(cfg, "train", domains)
    val_loader = build_dataloader(cfg, "val", domains)
    print(f"Train images: {len(train_loader.dataset)} | val images: {len(val_loader.dataset)}")

    hp = cfg["train"]
    if hp["init_checkpoint"]:
        model, _ = load_checkpoint(hp["init_checkpoint"], device)
    else:
        model = build_model(cfg["model"]["name"], len(classes), cfg["model"]["pretrained"])
        model = model.to(device)

    weights = None
    if hp["class_weights"]:
        weights = balanced_class_weights(train_loader.dataset.df["label"], classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=hp["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    freeze_epochs = hp["freeze_epochs"]
    if freeze_epochs:
        set_backbone_trainable(model, False)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg["output"]["runs_dir"]) / f"{cfg['experiment']}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    best_f1, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            print("Backbone unfrozen")

        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
            f"| val acc {val_acc:.4f} | val macro F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, stale_epochs = val_f1, 0
            save_checkpoint(checkpoint_path(cfg), model, cfg)
        else:
            stale_epochs += 1
            if stale_epochs >= hp["patience"]:
                print(f"Early stopping after epoch {epoch}")
                break

    print(f"Best val macro F1: {best_f1:.4f} | saved to {checkpoint_path(cfg)} | log: {run_dir}")
    return best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
