import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from waste_classifier.tracking import (  # noqa: E402
    SCENE_SIZE,
    WARM_UP_FRAMES,
    Detection,
    HandActivity,
    ItemTracker,
    SceneModel,
    follow_box,
    free_item,
    main_body,
    merge_boxes,
    overlap,
)

WIDTH, HEIGHT = 1280, 720
FRAME_SIZE = (WIDTH, HEIGHT)
NO_PERSON = np.zeros(SCENE_SIZE[::-1], np.uint8)
STEP = 1 / 30


def room(seed=0):
    """Return a textured room with a dark shelf object that's always there."""
    frame = cv2.resize(
        np.random.default_rng(seed).integers(110, 150, (45, 80, 3), dtype=np.uint8),
        FRAME_SIZE,
    )
    frame[100:260, 100:200] = (40, 60, 150)
    return frame


def with_item(frame, x, y, size=180):
    drawn = frame.copy()
    drawn[y : y + size, x : x + size] = (30, 200, 240)
    cv2.circle(drawn, (x + size // 2, y + size // 2), size // 4, (200, 40, 40), -1)
    return drawn


def run(tracker, frames, start=0.0, detections=(), hands=(), person=NO_PERSON):
    box = None
    for index, frame in enumerate(frames):
        box = tracker.update(frame, start + index * STEP, list(detections), list(hands), person)
    return box


def warmed_up(tracker, frames=WARM_UP_FRAMES + 10):
    run(tracker, [room()] * frames)
    return frames * STEP


def test_boxes_touching_are_merged_and_separate_ones_are_not():
    merged = merge_boxes([(0, 0, 10, 10), (9, 0, 20, 10), (100, 100, 110, 110)], margin=2)
    assert sorted(merged) == [(0, 0, 20, 10), (100, 100, 110, 110)]


def test_box_moves_at_once_but_changes_size_slowly():
    box = follow_box((0, 0, 100, 100), (200, 200, 500, 500))

    assert (box[0] + box[2]) / 2 == pytest.approx(350)
    assert box[2] - box[0] == pytest.approx(130)


def test_only_the_main_body_counts_as_a_person():
    person = NO_PERSON.copy()
    person[40:170, 100:180] = 255  # the body
    person[60:80, 260:280] = 255  # a held cup, marked as a person by mistake

    body = main_body(person)

    assert body[100, 140] == 255
    assert body[70, 270] == 0


def test_the_tracked_item_is_never_masked_as_a_person():
    person = NO_PERSON.copy()
    person[40:170, 100:180] = 255

    freed = free_item(person, (480, 200, 720, 680), FRAME_SIZE)

    assert freed[100, 140] == 0
    assert freed[100, 105] == 255


def test_a_resting_hand_is_not_active_but_a_moving_one_is():
    activity = HandActivity()
    resting = (300, 500, 380, 560)
    for frame in range(40):
        active = activity.active([resting], frame * STEP, WIDTH)
    assert active == []

    activity = HandActivity()
    for frame in range(40):
        moved = 300 + frame * 8
        active = activity.active([(moved, 500, moved + 80, 560)], frame * STEP, WIDTH)
    assert len(active) == 1


def test_swaying_with_the_body_does_not_make_a_hand_active():
    activity = HandActivity()
    for frame in range(40):
        moved = 300 + frame * 8
        body = np.array((moved - 100.0, 300.0))
        active = activity.active([(moved, 500, moved + 80, 560)], frame * STEP, WIDTH, body)

    assert active == []


def test_the_scene_ignores_what_was_there_and_reports_what_is_new():
    scene = SceneModel()
    for _ in range(WARM_UP_FRAMES + 5):
        changed = scene.update(room(), NO_PERSON, None)
    assert not changed.any()

    changed = scene.update(with_item(room(), 600, 300), NO_PERSON, None)

    assert changed[100, 170] > 0  # the new item
    assert changed[45, 37] == 0  # the shelf object


def test_a_brightness_change_is_not_a_new_item():
    scene = SceneModel()
    for _ in range(WARM_UP_FRAMES + 5):
        scene.update(room(), NO_PERSON, None)

    brighter = cv2.convertScaleAbs(room(), alpha=1, beta=25)
    changed = scene.update(brighter, NO_PERSON, None)

    assert (changed > 0).mean() < 0.02


def test_a_bumped_camera_is_not_a_new_item():
    scene = SceneModel()
    for _ in range(WARM_UP_FRAMES + 5):
        scene.update(room(), NO_PERSON, None)

    move = np.float32([[1, 0, 14], [0, 1, 9]])
    bumped = cv2.warpAffine(room(), move, FRAME_SIZE, borderMode=cv2.BORDER_REPLICATE)
    changed = scene.update(bumped, NO_PERSON, None)

    assert (changed > 0).mean() < 0.03


def test_the_wall_behind_a_person_who_moves_away_is_not_a_new_item():
    scene = SceneModel()
    body = NO_PERSON.copy()
    body[20:180, 120:200] = 255
    painted = room()
    painted[80:720, 480:800] = (60, 40, 170)  # the person, in front of the wall
    for _ in range(WARM_UP_FRAMES + 5):
        scene.update(painted, body, None)

    changed = scene.update(room(), NO_PERSON, None)

    assert not changed.any()


def test_an_empty_room_gets_no_box():
    tracker = ItemTracker()
    assert run(tracker, [room()] * 90) is None


def test_a_detected_object_that_was_already_in_the_room_is_ignored():
    tracker = ItemTracker()
    shelf = Detection((100, 100, 200, 260), 0.9, "bottle")

    assert run(tracker, [room()] * 90, detections=[shelf]) is None


def test_a_resting_hand_next_to_a_room_object_is_ignored():
    tracker = ItemTracker()
    table = Detection((300, 500, 900, 700), 0.4, "dining table")
    hand = (500, 520, 580, 570)

    assert run(tracker, [room()] * 90, detections=[table], hands=[hand]) is None


def test_a_new_moving_item_gets_locked_and_followed():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    frames = [with_item(room(), 300 + 12 * i, 300) for i in range(30)]

    box = run(tracker, frames, start)

    assert box is not None
    assert overlap(box, (648, 300, 828, 480)) > 0.4


def test_the_box_stays_on_an_item_that_stops_moving():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    moving = [with_item(room(), 300 + 12 * i, 300) for i in range(30)]
    run(tracker, moving, start)

    box = run(tracker, [with_item(room(), 648, 300)] * 240, start + 1)

    assert box is not None
    assert overlap(box, (648, 300, 828, 480)) > 0.4


def test_the_box_goes_away_with_the_item_and_leaves_no_ghost():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    moving = [with_item(room(), 300 + 12 * i, 300) for i in range(30)]
    run(tracker, moving, start)
    run(tracker, [with_item(room(), 648, 300)] * 150, start + 1)

    box = run(tracker, [room()] * 90, start + 6)

    assert box is None


def test_a_person_walking_by_is_not_an_item():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    frames, person = [], NO_PERSON.copy()
    person[50:170, 40:120] = 255
    for i in range(30):
        frames.append(with_item(room(), 200 + i * 2, 220, size=240))

    assert run(tracker, frames, start, person=person) is None


def test_a_picked_object_stays_locked_even_though_it_was_already_there():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    tracker.choose((100, 100, 200, 260), start)

    box = run(tracker, [room()] * 90, start)

    assert box is not None
    assert overlap(box, (100, 100, 200, 260)) > 0.5


def test_reset_forgets_the_item():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    tracker.choose((100, 100, 200, 260), start)

    tracker.reset()

    assert tracker.box is None
    assert run(tracker, [room()] * 5, start) is None


def test_each_new_item_gets_a_new_number():
    tracker = ItemTracker()
    start = warmed_up(tracker)
    tracker.choose((100, 100, 200, 260), start)
    first = tracker.item_number

    tracker.choose((400, 300, 600, 500), start + 1)

    assert tracker.item_number == first + 1
