import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

spec = importlib.util.spec_from_file_location("live_camera", Path("app/live_camera.py"))
live_camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_camera)

WIDTH, HEIGHT = 1280, 720


def test_summarize_averages_samples_so_one_bad_frame_does_not_win():
    samples = [
        {"Organic": 0.9, "Recyclable": 0.1},
        {"Organic": 0.8, "Recyclable": 0.2},
        {"Organic": 0.1, "Recyclable": 0.9},
    ]

    result = live_camera.summarize(samples, min_confidence=0.5)

    assert result.label == "Organic"
    assert result.confidence == pytest.approx(0.6)
    assert result.sure


def test_summarize_marks_low_confidence_results_as_unsure():
    result = live_camera.summarize([{"Organic": 0.55, "Recyclable": 0.45}], min_confidence=0.6)
    assert not result.sure


def test_summarize_without_samples_returns_nothing():
    assert live_camera.summarize([], min_confidence=0.6) is None


def test_only_models_on_disk_are_offered_with_the_best_first(tmp_path):
    (tmp_path / "basic_cnn.pt").touch()
    (tmp_path / "transfer_large.pt").touch()

    assert list(live_camera.find_models(tmp_path)) == ["Final model", "Model A"]


def test_missing_models_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained models"):
        live_camera.find_models(tmp_path)


def test_crop_is_square_and_stays_inside_the_frame():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    crop = live_camera.square_crop(frame, (0, 0, 100, 300))

    assert crop.shape[0] == crop.shape[1] == 390


def test_crop_of_a_huge_box_is_limited_to_the_frame():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    crop = live_camera.square_crop(frame, (0, 0, WIDTH, HEIGHT))

    assert crop.shape[:2] == (HEIGHT, HEIGHT)


def test_filter_removes_jitter_from_a_still_item():
    rng = np.random.default_rng(0)
    box_filter = live_camera.BoxFilter()
    shown = []
    for i in range(120):
        noise = rng.normal(0, 4, 4)
        shown.append(box_filter(tuple(np.array((400, 200, 700, 500)) + noise), i / 30))

    steps = np.abs(np.diff(np.array(shown[30:]), axis=0))
    assert steps.mean() < 1


def test_filter_keeps_up_with_a_moving_item():
    box_filter = live_camera.BoxFilter()
    for i in range(60):
        x = 100 + 10 * i
        shown = box_filter((x, 200, x + 300, 500), i / 30)

    assert abs(shown[0] - x) < 30


def test_filter_jumps_straight_to_a_new_item():
    box_filter = live_camera.BoxFilter()
    box_filter((0, 0, 100, 100), 0)

    assert box_filter((800, 400, 900, 500), 0.03) == (800, 400, 900, 500)


@pytest.mark.parametrize("height,width", [(720, 1280), (1080, 1920), (480, 640)])
@pytest.mark.parametrize(
    "scan",
    [
        live_camera.Scan(),
        live_camera.Scan(box=(300, 200, 500, 400)),
        live_camera.Scan(
            box=(300, 200, 500, 400), result=live_camera.Result("Organic", 0.5, False)
        ),
        live_camera.Scan(
            box=(0, 0, 639, 479),
            result=live_camera.Result("Non-Recyclable", 0.97, True),
        ),
    ],
)
def test_overlay_works_at_common_camera_sizes(scan, height, width):
    frame = np.full((height, width, 3), 90, dtype=np.uint8)

    drawn = live_camera.draw_overlay(frame, scan, "Fine-tuned Model B")

    assert drawn.shape == frame.shape
    assert (drawn != frame).any()
