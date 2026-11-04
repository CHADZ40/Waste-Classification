"""Build the data manifest: one row per image with its class, image type and split.

Splitting is stratified by source category and image type, so every category keeps the
same train/val/test proportions in both default and real_world images. Byte-identical
files are removed first so the same image cannot appear in two splits.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from waste_classifier.utils import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DOMAINS = ("default", "real_world")
EXPECTED_PER_FOLDER = 250


def find_dataset_root(raw_dir: Path, category: str) -> Path:
    """Return the folder holding the category folders, which may be nested after unzipping."""
    if (raw_dir / category).is_dir():
        return raw_dir
    for match in raw_dir.rglob(category):
        if match.is_dir():
            return match.parent
    raise FileNotFoundError(f"Could not find a '{category}' folder under {raw_dir}")


def collect_images(raw_dir: Path, mapping: dict[str, list[str]]) -> pd.DataFrame:
    root = find_dataset_root(raw_dir, next(iter(mapping.values()))[0])
    rows = []
    for label, categories in mapping.items():
        for category in categories:
            for domain in DOMAINS:
                folder = root / category / domain
                images = sorted(
                    p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not images:
                    raise FileNotFoundError(f"No images found in {folder}")
                if len(images) != EXPECTED_PER_FOLDER:
                    print(f"Warning: {folder} has {len(images)} images")
                rows += [
                    {
                        "path": p.relative_to(raw_dir).as_posix(),
                        "label": label,
                        "category": category,
                        "domain": domain,
                        "hash": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in images
                ]
    return pd.DataFrame(rows)


def make_manifest(
    raw_dir: Path,
    manifest_path: Path,
    mapping: dict[str, list[str]],
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    df = collect_images(raw_dir, mapping)
    total = len(df)
    df = df.drop_duplicates("hash").drop(columns="hash").reset_index(drop=True)
    print(f"Found {total} images, removed {total - len(df)} exact duplicates")

    strata = df["category"] + "/" + df["domain"]
    train_val, test = train_test_split(
        df.index, test_size=test_size, stratify=strata, random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=strata[train_val],
        random_state=seed,
    )
    df.loc[train, "split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    with open(data["class_mapping"]) as f:
        mapping = yaml.safe_load(f)

    df = make_manifest(
        Path(data["raw_dir"]),
        Path(data["manifest"]),
        mapping,
        data["val_size"],
        data["test_size"],
        cfg["seed"],
    )
    print(df.groupby(["split", "domain", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
