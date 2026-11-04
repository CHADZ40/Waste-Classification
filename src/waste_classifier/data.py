"""Dataset, transforms and dataloaders."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> T.Compose:
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                *normalize,
            ]
        )
    return T.Compose([T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size), *normalize])


def load_subset(cfg: dict, split: str, domains: list[str]) -> pd.DataFrame:
    """Return the manifest rows for one split, limited to the given image types."""
    df = pd.read_csv(cfg["data"]["manifest"])
    subset = df[(df["split"] == split) & df["domain"].isin(domains)]
    if subset.empty:
        raise ValueError(f"No images for split={split!r}, domains={domains}")
    return subset.reset_index(drop=True)


class WasteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, raw_dir: Path, classes: list[str], transform=None):
        self.df = df
        self.raw_dir = Path(raw_dir)
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        with Image.open(self.raw_dir / row["path"]) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[row["label"]]


def build_dataloader(cfg: dict, split: str, domains: list[str]) -> DataLoader:
    data = cfg["data"]
    train = split == "train"
    dataset = WasteDataset(
        load_subset(cfg, split, domains),
        data["raw_dir"],
        data["classes"],
        build_transforms(data["image_size"], train),
    )
    return DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=train,
        num_workers=data["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
