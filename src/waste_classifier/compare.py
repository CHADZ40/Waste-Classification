"""Collect every evaluated experiment into reports/comparison.csv."""

import argparse
import json
from pathlib import Path

import pandas as pd


def compare(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(reports_dir.glob("*/metrics.json")):
        results = json.loads(path.read_text())
        default, real_world = results["default"], results["real_world"]
        real_world_all = results.get("real_world_all", {})
        gap = results.get("gap_all", results["gap"])
        rows.append(
            {
                "experiment": path.parent.name,
                "default_macro_f1": default["macro_f1"],
                "real_world_all_macro_f1": real_world_all.get("macro_f1"),
                "real_world_test_macro_f1": real_world["macro_f1"],
                "default_acc": default["accuracy"],
                "real_world_all_acc": real_world_all.get("accuracy"),
                "real_world_test_acc": real_world["accuracy"],
                "macro_f1_drop": gap["macro_f1_drop"],
                "most_affected_class": next(iter(gap["per_class_f1_drop"])),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.json files under {reports_dir}")
    table = pd.DataFrame(rows).round(4)
    table.to_csv(reports_dir / "comparison.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    print(compare(Path(args.reports_dir)).to_string(index=False))


if __name__ == "__main__":
    main()
