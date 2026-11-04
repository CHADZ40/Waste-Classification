import pytest

from waste_classifier.evaluate import compute_gap, compute_metrics

CLASSES = ["a", "b"]


def test_metrics_score_a_class_that_is_never_predicted_as_zero():
    metrics = compute_metrics([0, 0, 0, 1], [0, 0, 0, 0], CLASSES)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["per_class"]["b"]["f1"] == 0
    assert metrics["macro_f1"] == pytest.approx((6 / 7) / 2)


def test_gap_lists_the_most_affected_class_first():
    default = compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], CLASSES)
    real_world = compute_metrics([0, 0, 1, 1], [0, 0, 0, 1], CLASSES)

    gap = compute_gap(default, real_world)

    assert gap["macro_f1_drop"] > 0
    assert list(gap["per_class_f1_drop"]) == ["b", "a"]
