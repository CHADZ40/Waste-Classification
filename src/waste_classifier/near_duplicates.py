"""Flag test images that are near-copies of images a model could have trained on.

Exact copies are removed by split.py, but the dataset also has resized and re-encoded
copies. This adds two columns to the manifest without changing any split:

- near_studio: close to a default train or val image
- near_any: close to a default or real_world train or val image

Images are compared as 32x32 color thumbnails. A perceptual hash was tried first, but it
matched unrelated items that share a white background.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from waste_classifier.utils import load_config

THUMBNAIL_SIZE = 32


def load_thumbnails(paths: pd.Series, raw_dir: Path) -> np.ndarray:
    thumbnails = []
    for path in tqdm(paths, desc="thumbnails"):
        with Image.open(raw_dir / path) as image:
            small = image.convert("RGB").resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.BILINEAR)
        thumbnails.append(np.asarray(small, dtype=np.float32).ravel() / 255)
    return np.stack(thumbnails)


def nearest_rmse(queries: np.ndarray, pool: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Return each query's root-mean-square pixel difference to its closest pool image."""
    pool_norms = (pool**2).sum(axis=1)
    distances = []
    for start in range(0, len(queries), chunk):
        batch = queries[start : start + chunk]
        squared = (batch**2).sum(axis=1)[:, None] + pool_norms[None, :] - 2 * batch @ pool.T
        distances.append(np.maximum(squared.min(axis=1), 0))
    return np.sqrt(np.concatenate(distances) / queries.shape[1])


def flag_near_duplicates(
    df: pd.DataFrame, thumbnails: np.ndarray, threshold: float
) -> pd.DataFrame:
    df = df.copy()
    not_test = df["split"] != "test"
    studio_pool = not_test & (df["domain"] == "default")

    candidates = ~studio_pool
    df["near_studio"] = False
    df.loc[candidates, "near_studio"] = (
        nearest_rmse(thumbnails[candidates], thumbnails[studio_pool]) < threshold
    )

    is_test = ~not_test
    df["near_any"] = False
    df.loc[is_test, "near_any"] = (
        nearest_rmse(thumbnails[is_test], thumbnails[not_test]) < threshold
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    data = load_config(args.config)["data"]
    manifest = Path(data["manifest"])
    df = pd.read_csv(manifest)
    thumbnails = load_thumbnails(df["path"], Path(data["raw_dir"]))
    df = flag_near_duplicates(df, thumbnails, data["near_duplicate_rmse"])
    df.to_csv(manifest, index=False)

    test = df[df["split"] == "test"]
    print(f"Flagged {df['near_studio'].sum()} images near studio train/val images")
    print(f"Flagged {test['near_any'].sum()} of {len(test)} test images near any train/val image")


if __name__ == "__main__":
    main()
