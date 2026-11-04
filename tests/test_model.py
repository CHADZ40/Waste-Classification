import pytest
import torch
from PIL import Image

from waste_classifier.data import build_transforms
from waste_classifier.model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)


@pytest.mark.parametrize("name", ["basic_cnn", "resnet18"])
def test_model_outputs_one_logit_per_class(name):
    model = build_model(name, num_classes=4, pretrained=False).eval()
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 4)


def test_freezing_leaves_only_the_head_trainable():
    model = build_model("resnet18", num_classes=4, pretrained=False)
    set_backbone_trainable(model, False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}

    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_checkpoint_round_trip(tmp_path):
    cfg = {
        "model": {"name": "basic_cnn"},
        "data": {"classes": ["Organic", "Recyclable"], "image_size": 64},
    }
    model = build_model("basic_cnn", num_classes=2, pretrained=False).eval()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, cfg)

    loaded, meta = load_checkpoint(path, torch.device("cpu"))

    x = torch.randn(1, 3, 64, 64)
    assert torch.equal(model(x), loaded(x))
    assert meta["classes"] == ["Organic", "Recyclable"]


def test_transforms_produce_requested_size():
    image = Image.new("RGB", (256, 256))
    for train in (True, False):
        assert build_transforms(64, train)(image).shape == (3, 64, 64)
