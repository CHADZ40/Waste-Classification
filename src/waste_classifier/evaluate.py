"""Score a trained experiment on its test sets.

- default: held-out studio images.
- real_world: held-out real_world images, comparable across every experiment (RQ3).
- real_world_all: every real_world image, for models that never trained on them (RQ1, RQ2).

Near-copies of images a model could have trained on are left out of every test set.

Writes metrics.json, predictions.csv and one confusion matrix per test set to
reports/<experiment>/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

from waste_classifier.data import build_dataloader, load_subset
from waste_classifier.engine import run_epoch
from waste_classifier.model import load_checkpoint
from waste_classifier.utils import checkpoint_path, get_device, load_config


def build_test_sets(cfg: dict) -> dict[str, tuple[str, list[str], str]]:
    """Map each test set name to its split, image types and near-copy exclusion column."""
    sets = {
        "default": ("test", ["default"], "near_any"),
        "real_world": ("test", ["real_world"], "near_any"),
    }
    if "real_world" not in cfg["data"]["train_domains"]:
        sets["real_world_all"] = ("all", ["real_world"], "near_studio")
    return sets


def compute_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
    }


def compute_gap(default: dict, real_world: dict) -> dict:
    """Default minus real_world scores, with the most affected class first."""
    per_class = {
        name: default["per_class"][name]["f1"] - real_world["per_class"][name]["f1"]
        for name in default["per_class"]
    }
    return {
        "macro_f1_drop": default["macro_f1"] - real_world["macro_f1"],
        "per_class_f1_drop": dict(sorted(per_class.items(), key=lambda x: x[1], reverse=True)),
    }


def save_confusion_matrix(y_true, y_pred, classes: list[str], path: Path) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        display_labels=classes,
        normalize="true",
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45,
    )
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=200)
    plt.close(display.figure_)


def evaluate(cfg: dict) -> dict:
    device = get_device()
    model, meta = load_checkpoint(checkpoint_path(cfg), device)
    classes = meta["classes"]
    report_dir = Path(cfg["output"]["reports_dir"]) / cfg["experiment"]
    report_dir.mkdir(parents=True, exist_ok=True)

    train_labels = load_subset(cfg, "train", cfg["data"]["train_domains"])["label"]
    majority = classes.index(train_labels.mode()[0])

    results, predictions = {}, []
    sets = build_test_sets(cfg)
    for name, (split, domains, exclude) in sets.items():
        loader = build_dataloader(cfg, split, domains, exclude)
        _, y_true, y_pred = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
        results[name] = compute_metrics(y_true, y_pred, classes)
        results[name]["majority_baseline"] = compute_metrics(
            y_true, [majority] * len(y_true), classes
        )
        save_confusion_matrix(y_true, y_pred, classes, report_dir / f"confusion_{name}.png")
        predictions.append(
            loader.dataset.df.assign(test_set=name, pred=[classes[i] for i in y_pred])
        )

    results["gap"] = compute_gap(results["default"], results["real_world"])
    if "real_world_all" in sets:
        results["gap_all"] = compute_gap(results["default"], results["real_world_all"])
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.concat(predictions).to_csv(report_dir / "predictions.csv", index=False)

    for name in sets:
        scores = results[name]
        print(
            f"{name:<15} macro F1 {scores['macro_f1']:.4f} "
            f"| balanced acc {scores['balanced_accuracy']:.4f} "
            f"| acc {scores['accuracy']:.4f} "
            f"| majority macro F1 {scores['majority_baseline']['macro_f1']:.4f}"
        )
    gap = results.get("gap_all", results["gap"])
    print(f"Macro F1 drop: {gap['macro_f1_drop']:.4f} | saved to {report_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
