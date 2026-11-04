import pandas as pd
import pytest
import yaml

from waste_classifier.train import balanced_class_weights
from waste_classifier.utils import load_config


def test_class_mapping_matches_the_proposal():
    with open("configs/class_mapping.yaml") as f:
        mapping = yaml.safe_load(f)

    sizes = {label: len(categories) for label, categories in mapping.items()}
    assert sizes == {"Hazardous": 1, "Non-Recyclable": 8, "Organic": 4, "Recyclable": 17}
    categories = [c for group in mapping.values() for c in group]
    assert len(set(categories)) == 30


def test_experiment_configs_inherit_and_override():
    cfg = load_config("configs/transfer_real_world.yaml")

    assert cfg["experiment"] == "transfer_real_world"
    assert cfg["model"]["name"] == "efficientnet_b0"
    assert cfg["data"]["train_domains"] == ["default", "real_world"]
    assert cfg["data"]["image_size"] == 224
    assert cfg["data"]["classes"][0] == "Hazardous"


def test_class_weights_favor_rare_classes():
    labels = pd.Series(["a"] * 30 + ["b"] * 10)
    weights = balanced_class_weights(labels, ["a", "b"])
    assert weights[1] == pytest.approx(3 * weights[0])


def test_class_weight_power_softens_the_weighting():
    labels = pd.Series(["a"] * 90 + ["b"] * 10)
    softened = balanced_class_weights(labels, ["a", "b"], power=0.5)
    assert softened[1] == pytest.approx(3 * softened[0])

    unweighted = balanced_class_weights(labels, ["a", "b"], power=0)
    assert unweighted.tolist() == [1, 1]


def test_class_weights_reject_a_missing_class():
    with pytest.raises(ValueError, match="b"):
        balanced_class_weights(pd.Series(["a", "a"]), ["a", "b"])
