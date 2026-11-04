import shutil

from PIL import Image

from waste_classifier.split import DOMAINS, make_manifest

MAPPING = {"Organic": ["eggshells"], "Recyclable": ["newspaper", "shoes"]}


def make_dataset(raw_dir, per_folder=20):
    color = 0
    for categories in MAPPING.values():
        for category in categories:
            for domain in DOMAINS:
                folder = raw_dir / "images" / "images" / category / domain
                folder.mkdir(parents=True)
                for i in range(per_folder):
                    color += 1
                    Image.new("RGB", (8, 8), (color % 256, color // 256, 0)).save(
                        folder / f"Image_{i}.png"
                    )


def test_manifest_splits_every_category_and_image_type_evenly(tmp_path):
    make_dataset(tmp_path)

    df = make_manifest(tmp_path, tmp_path / "manifest.csv", MAPPING, 0.15, 0.15, seed=0)

    assert len(df) == 120
    assert df["path"].is_unique
    assert (df["path"].str.startswith("images/images/")).all()
    counts = df.groupby(["category", "domain", "split"]).size()
    assert (counts.xs("test", level="split") == 3).all()
    assert (counts.xs("val", level="split") == 3).all()
    assert (counts.xs("train", level="split") == 14).all()


def test_identical_files_are_removed_so_they_cannot_cross_splits(tmp_path):
    make_dataset(tmp_path)
    root = tmp_path / "images" / "images"
    shutil.copy(root / "shoes/default/Image_0.png", root / "shoes/real_world/Image_0.png")

    df = make_manifest(tmp_path, tmp_path / "manifest.csv", MAPPING, 0.15, 0.15, seed=0)

    assert len(df) == 119
