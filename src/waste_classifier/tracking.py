"""Decide which item the camera should follow.

A detector alone can't tell the item someone is showing from a cup on a shelf. This module
adds what's missing:

- SceneModel learns the room, so things that were already there are ignored.
- ItemTracker only accepts something new or held, needs it to last for a moment, then locks
  on and follows it with optical flow until it leaves.
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]

SCENE_SIZE = (320, 180)
WARM_UP_FRAMES = 20
CHANGE_THRESHOLD = 20.0
LEARN_BACKGROUND = 0.03
LEARN_PERSON = 0.05
LEARN_CHANGED = 0.002
LEARN_STILL_CHANGE = 0.08
STILL_CHANGE_FRAMES = 40
MIN_CAMERA_SHIFT = 1.0
MIN_SHIFT_RESPONSE = 0.15
RESET_SHARE = 0.55
RESET_FRAMES = 30
MIN_REGION_SHARE = 0.006
MAX_REGION_SHARE = 0.5
PERSON_MARGIN = 3
MAIN_BODY_SHARE = 0.25
LEARNING_MARGIN = 12  # segmentation runs small, so stay clear of a body while learning the room
HUG_MARGIN = 8  # a new region within this distance of a body may just be the body moving
MAX_SLIVER_SHARE = 0.025  # only regions smaller than this can be dismissed as a moving body edge
HAND_ZONE = 1.0  # how far, in hand sizes, around a hand doesn't count as the person

MIN_SCORE = 0.3
MIN_NEW_SHARE = 0.25
MIN_ITEM_DIFFERENCE = 0.15  # a tracked box must differ this much from the empty room
LOCK_FRAMES = 6
LOST_SECONDS = 0.45
MOTION_THRESHOLD = 12
MIN_MOTION_SHARE = 0.04
HAND_MARGIN = 0.03
MIN_FLOW_POINTS = 4
MIN_HELD_NEW_SHARE = 0.1
MAX_HELD_SIZE = 14  # a held item's area, compared with the hand's
MERGE_MARGIN = 0.015
SWITCH_FRAMES = 15
MIN_RIVAL_SIZE = 0.6  # an unheld rival must be at least this big, compared with the item
SIZE_RATE = 0.15
BODY_SHARE = 0.65  # a region mostly covered by a body outline is a person, not an item
BODY_FRAMES = 15
HAND_MEMORY_SECONDS = 1.0
HAND_ACTIVE_DISTANCE = 0.035  # share of the frame width a hand must travel to count as active

# Objects that are nearly always the room, so they need a hand or movement to count.
ROOM_OBJECTS = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "keyboard",
    "mouse",
    "potted plant",
    "clock",
    "refrigerator",
    "oven",
    "microwave",
    "sink",
    "toilet",
    "bench",
}


@dataclass
class Detection:
    box: Box
    score: float
    name: str


@dataclass
class Candidate:
    box: Box
    held: bool
    source: str


def overlap(a: Box, b: Box) -> float:
    """Return the intersection over union of two boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


def near(a: Box, b: Box, margin: float) -> bool:
    return (
        a[0] - margin < b[2]
        and b[0] - margin < a[2]
        and a[1] - margin < b[3]
        and b[1] - margin < a[3]
    )


def main_body(person: np.ndarray) -> np.ndarray:
    """Keep the large parts of a person mask.

    Selfie segmentation often marks a separate item, such as a held cup, as a second person.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(person)
    if count <= 2:
        return person
    largest = stats[1:, 4].max()
    keep = [i for i in range(1, count) if stats[i, 4] >= MAIN_BODY_SHARE * largest]
    return np.isin(labels, keep).astype(np.uint8) * 255


def free_zones(
    person: np.ndarray, hands: list[Box], item: Box | None, frame_size: tuple[int, int]
) -> np.ndarray:
    """Return a person mask without the tracked item or the area around active hands.

    Selfie segmentation tends to count whatever someone holds as part of them.
    """
    freed = main_body(person)
    scale_x, scale_y = SCENE_SIZE[0] / frame_size[0], SCENE_SIZE[1] / frame_size[1]
    zones = [
        (x0 - (x1 - x0) * HAND_ZONE, y0 - (y1 - y0) * HAND_ZONE,
         x1 + (x1 - x0) * HAND_ZONE, y1 + (y1 - y0) * HAND_ZONE)
        for x0, y0, x1, y1 in hands
    ]
    if item is not None:
        zones.append(item)
    if not zones:
        return freed
    freed = freed.copy()
    for x0, y0, x1, y1 in zones:
        left, top = max(int(x0 * scale_x), 0), max(int(y0 * scale_y), 0)
        freed[top : int(y1 * scale_y) + 1, left : int(x1 * scale_x) + 1] = 0
    return freed


def merge_boxes(boxes: list[Box], margin: float) -> list[Box]:
    """Join boxes that touch, so the pieces of one item become one box."""
    merged = list(boxes)
    joined = True
    while joined:
        joined = False
        for i, first in enumerate(merged):
            for j in range(i + 1, len(merged)):
                second = merged[j]
                if near(first, second, margin):
                    merged[i] = (
                        min(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[2], second[2]),
                        max(first[3], second[3]),
                    )
                    del merged[j]
                    joined = True
                    break
            if joined:
                break
    return merged


def follow_box(current: Box, target: Box) -> Box:
    """Move to a target's center at once, but change size slowly.

    Regions around a held item grow and shrink a lot, while the item itself doesn't.
    """
    center_x, center_y = (target[0] + target[2]) / 2, (target[1] + target[3]) / 2
    width = current[2] - current[0]
    height = current[3] - current[1]
    width += SIZE_RATE * (target[2] - target[0] - width)
    height += SIZE_RATE * (target[3] - target[1] - height)
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def box_area(box: Box) -> float:
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def share_inside(mask: np.ndarray, box: Box, scale_x: float, scale_y: float) -> float:
    """Return the share of a box, given in frame pixels, that is set in a small mask."""
    x0, y0 = max(int(box[0] * scale_x), 0), max(int(box[1] * scale_y), 0)
    x1, y1 = int(box[2] * scale_x) + 1, int(box[3] * scale_y) + 1
    region = mask[y0:y1, x0:x1]
    return float((region > 0).mean()) if region.size else 0.0


class HandActivity:
    """Separates a hand showing something from one resting on a desk or lap."""

    def __init__(self):
        self.history = deque()

    def active(self, hands: list[Box], now: float, width: int) -> list[Box]:
        while self.history and now - self.history[0][0] > HAND_MEMORY_SECONDS:
            self.history.popleft()
        limit = HAND_ACTIVE_DISTANCE * width
        result = []
        for hand in hands:
            center = np.array(((hand[0] + hand[2]) / 2, (hand[1] + hand[3]) / 2))
            past = [
                np.linalg.norm(center - point) for _, points in self.history for point in points
            ]
            # The farthest nearby position it recently held, ignoring other hands far away.
            local = [distance for distance in past if distance < 6 * limit]
            if not local or max(local) >= limit:
                result.append(hand)
        points = [np.array(((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)) for h in hands]
        self.history.append((now, points))
        return result


class SceneModel:
    """Learns what the room looks like and reports what has changed."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.background = None
        self.snapshot = None
        self.snapshot_offset = 0.0
        self.latest = None
        self.previous = None
        self.moving = None
        self.still_change = None
        self.seen = None
        self.frames = 0
        self.busy_frames = 0

    def follow_camera(self, lab: np.ndarray) -> None:
        """Realign the learned room if the camera got bumped."""
        if self.background is None or self.frames <= WARM_UP_FRAMES:
            return
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            np.ascontiguousarray(self.background[:, :, 0]), np.ascontiguousarray(lab[:, :, 0])
        )
        if response < MIN_SHIFT_RESPONSE or np.hypot(shift_x, shift_y) < MIN_CAMERA_SHIFT:
            return
        move = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        size = SCENE_SIZE
        self.background = cv2.warpAffine(
            self.background, move, size, borderMode=cv2.BORDER_REPLICATE
        )
        if self.snapshot is not None:
            self.snapshot = cv2.warpAffine(
                self.snapshot, move, size, borderMode=cv2.BORDER_REPLICATE
            )

    def take_snapshot(self) -> None:
        """Remember the room as it was before an item arrived."""
        self.snapshot = None if self.background is None else self.background.copy()

    def restore_snapshot(self) -> None:
        """Undo anything learned from the item, so it can't leave a ghost behind."""
        if self.snapshot is not None:
            self.background = self.snapshot
        self.snapshot = None
        if self.still_change is not None:
            self.still_change[:] = 0

    def differs_from_snapshot(self, box: Box, frame_size: tuple[int, int]) -> float:
        """Return the share of a box that doesn't look like the empty room."""
        if self.snapshot is None or self.latest is None:
            return 1.0
        scale_x, scale_y = SCENE_SIZE[0] / frame_size[0], SCENE_SIZE[1] / frame_size[1]
        x0, y0 = max(int(box[0] * scale_x), 0), max(int(box[1] * scale_y), 0)
        x1, y1 = int(box[2] * scale_x) + 1, int(box[3] * scale_y) + 1
        difference = (self.latest - self.snapshot)[y0:y1, x0:x1]
        if difference.size == 0:
            return 0.0
        difference = difference.copy()
        difference[:, :, 0] -= self.snapshot_offset
        change = np.abs(difference[:, :, 0]) * 0.5 + np.linalg.norm(difference[:, :, 1:], axis=2)
        return float((change > CHANGE_THRESHOLD).mean())

    def update(self, frame: np.ndarray, person: np.ndarray, protected: Box | None) -> np.ndarray:
        """Return a mask of new things. `protected` is a box, in frame pixels, to not learn."""
        small = cv2.resize(frame, SCENE_SIZE, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)
        self.latest = lab
        self.follow_camera(lab)
        if self.background is None:
            self.background = lab
            self.previous = lab
            self.still_change = np.zeros(lab.shape[:2], np.uint16)
            self.seen = np.zeros(lab.shape[:2], bool)
        self.frames += 1
        motion = np.abs(lab[:, :, 0] - self.previous[:, :, 0]) > MOTION_THRESHOLD
        self.moving = motion.astype(np.uint8)
        self.previous = lab

        near_person = cv2.dilate(person, np.ones((2 * PERSON_MARGIN + 1,) * 2, np.uint8)) > 0
        # What's behind a person isn't known yet. When it's revealed, it's room, not a new item.
        revealed = ~near_person & ~self.seen
        self.background[revealed] = lab[revealed]
        if self.snapshot is not None:
            self.snapshot[revealed] = lab[revealed]
        if self.frames <= WARM_UP_FRAMES:
            wide = cv2.dilate(person, np.ones((2 * LEARNING_MARGIN + 1,) * 2, np.uint8))
            self.seen = wide == 0
        else:
            self.seen |= ~near_person

        # Webcams change exposure on their own, so the overall brightness doesn't count.
        calm = ~near_person
        offset = float(np.median((lab - self.background)[calm][:, 0])) if calm.any() else 0.0
        if self.snapshot is not None and calm.any():
            self.snapshot_offset = float(np.median((lab - self.snapshot)[calm][:, 0]))
        difference = lab - self.background
        difference[:, :, 0] -= offset
        change = np.abs(difference[:, :, 0]) * 0.5 + np.linalg.norm(difference[:, :, 1:], axis=2)
        changed = (change > CHANGE_THRESHOLD).astype(np.uint8) * 255
        raw_change = changed > 0
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        changed[near_person] = 0

        if self.frames <= WARM_UP_FRAMES:
            self.background += 0.2 * (lab - self.background)
            return np.zeros(changed.shape, np.uint8)

        # Something changed and has stopped moving: an abandoned ghost, not an item.
        still = (changed > 0) & ~motion
        self.still_change = np.where(still, self.still_change + 1, 0).astype(np.uint16)

        rate = np.where(changed > 0, LEARN_CHANGED, LEARN_BACKGROUND).astype(np.float32)
        rate[self.still_change >= STILL_CHANGE_FRAMES] = LEARN_STILL_CHANGE
        # A person's outline changes all the time, but a change there may be a held item.
        rate[near_person & ~raw_change] = LEARN_PERSON
        rate[near_person & raw_change] = LEARN_CHANGED
        if protected is not None:
            # Protect everything connected to the item, not just the box around it.
            scale_x, scale_y = SCENE_SIZE[0] / frame.shape[1], SCENE_SIZE[1] / frame.shape[0]
            x0, y0 = max(int(protected[0] * scale_x), 0), max(int(protected[1] * scale_y), 0)
            x1, y1 = int(protected[2] * scale_x) + 1, int(protected[3] * scale_y) + 1
            _, labels = cv2.connectedComponents(changed)
            touching = np.unique(labels[y0:y1, x0:x1])
            rate[np.isin(labels, touching[touching > 0])] = 0
            rate[y0:y1, x0:x1] = 0
            self.still_change[rate == 0] = 0
        self.background += rate[:, :, None] * (lab - self.background)

        # If most of the view changed, the camera probably moved.
        self.busy_frames = self.busy_frames + 1 if (changed > 0).mean() > RESET_SHARE else 0
        if self.busy_frames >= RESET_FRAMES:
            self.reset()
        return changed


class ItemTracker:
    """Chooses one item, locks on and follows it until it leaves."""

    def __init__(self):
        self.scene = SceneModel()
        self.hand_activity = HandActivity()
        self.item_number = 0
        self.clear()

    def clear(self) -> None:
        self.box = None
        self.manual = False
        self.held = False
        self.pending = None
        self.pending_frames = 0
        self.rival = None
        self.rival_frames = 0
        self.body = None
        self.body_frames = 0
        self.last_support = 0.0
        self.points = None
        self.gray = None

    def reset(self) -> None:
        """Forget the item and relearn the room."""
        self.clear()
        self.scene.reset()

    def choose(self, box: Box, now: float) -> None:
        """Follow something the user picked until they reset, whatever it looks like."""
        self.clear_item()
        self.lock(box, now, manual=True)
        self.scene.snapshot = None

    def lock(self, box: Box, now: float, manual: bool = False) -> None:
        if self.box is None:
            self.scene.take_snapshot()
            self.item_number += 1
        self.box, self.manual, self.last_support = box, manual, now
        self.pending, self.pending_frames, self.points = None, 0, None

    def find_regions(self, changed: np.ndarray, frame_size: tuple[int, int]) -> list[Box]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(changed)
        total = changed.size
        ratio_x, ratio_y = frame_size[0] / SCENE_SIZE[0], frame_size[1] / SCENE_SIZE[1]
        pieces = [
            (x * ratio_x, y * ratio_y, (x + w) * ratio_x, (y + h) * ratio_y)
            for x, y, w, h, area in stats[1:count]
            if area / total >= MIN_REGION_SHARE / 4
        ]
        limit = frame_size[0] * frame_size[1]
        return [
            box
            for box in merge_boxes(pieces, MERGE_MARGIN * frame_size[0])
            if MIN_REGION_SHARE * limit <= box_area(box) <= MAX_REGION_SHARE * limit
        ]

    def find_candidates(
        self,
        detections: list[Detection],
        regions: list[Box],
        hands: list[Box],
        changed: np.ndarray,
        moving: np.ndarray,
        person: np.ndarray,
        frame_size: tuple[int, int],
    ) -> list[Candidate]:
        width, height = frame_size
        scale_x, scale_y = SCENE_SIZE[0] / width, SCENE_SIZE[1] / height
        margin = HAND_MARGIN * width

        def is_held(box: Box) -> bool:
            """A hand alone isn't enough: resting hands sit next to all sorts of things."""
            if share_inside(changed, box, scale_x, scale_y) < MIN_HELD_NEW_SHARE:
                return False
            return any(
                near(box, hand, margin) and box_area(box) <= MAX_HELD_SIZE * box_area(hand)
                for hand in hands
            )

        found = []
        for item in detections:
            if item.score < MIN_SCORE or item.name == "person":
                continue
            area = (item.box[2] - item.box[0]) * (item.box[3] - item.box[1])
            if area > MAX_REGION_SHARE * width * height:
                continue
            held = item.name not in ROOM_OBJECTS and is_held(item.box)
            is_new = share_inside(changed, item.box, scale_x, scale_y) >= MIN_NEW_SHARE
            on_person = share_inside(person, item.box, scale_x, scale_y) > 0.5
            if held or (is_new and not on_person and item.name not in ROOM_OBJECTS):
                found.append(Candidate(item.box, held, "detector"))

        hugging = cv2.dilate(self.body, np.ones((2 * HUG_MARGIN + 1,) * 2, np.uint8))
        for region in regions:
            if any(overlap(region, c.box) > 0.3 for c in found):
                continue
            held = is_held(region)
            active = share_inside(moving, region, scale_x, scale_y) >= MIN_MOTION_SHARE
            small = box_area(region) <= MAX_SLIVER_SHARE * width * height
            outline = hugging if small else self.body
            is_body = share_inside(outline, region, scale_x, scale_y) >= BODY_SHARE
            if held or (active and not is_body):
                found.append(Candidate(region, held, "new"))
        return found

    def follow(self, gray: np.ndarray) -> None:
        """Move the box with the points inside it, so it survives gaps in detection."""
        if self.gray is None or self.box is None:
            return
        if self.points is None or len(self.points) < MIN_FLOW_POINTS:
            region = np.zeros(gray.shape, np.uint8)
            x0, y0 = max(int(self.box[0] / 2), 0), max(int(self.box[1] / 2), 0)
            region[y0 : int(self.box[3] / 2), x0 : int(self.box[2] / 2)] = 255
            self.points = cv2.goodFeaturesToTrack(
                self.gray, maxCorners=40, qualityLevel=0.02, minDistance=5, mask=region
            )
        if self.points is None:
            return
        moved, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
        if moved is None:
            self.points = None
            return
        good = status.ravel() == 1
        if good.sum() < MIN_FLOW_POINTS:
            self.points = None
            return
        shift = np.median((moved[good] - self.points[good]).reshape(-1, 2), axis=0) * 2
        self.box = (
            self.box[0] + shift[0],
            self.box[1] + shift[1],
            self.box[2] + shift[0],
            self.box[3] + shift[1],
        )
        self.points = moved[good].reshape(-1, 1, 2)

    def update(
        self,
        frame: np.ndarray,
        now: float,
        detections: list[Detection],
        hands: list[Box],
        person: np.ndarray,
    ) -> Box | None:
        """Return the box of the tracked item, or None if nothing is being shown."""
        height, width = frame.shape[:2]
        size = (width, height)
        hands = self.hand_activity.active(hands, now, width)
        self.body = main_body(person)
        person = free_zones(person, hands, self.box, size)
        changed = self.scene.update(frame, person, self.box)
        moving = self.scene.moving

        half = cv2.cvtColor(cv2.resize(frame, (width // 2, height // 2)), cv2.COLOR_BGR2GRAY)
        self.follow(half)
        self.gray = half

        regions = self.find_regions(changed, size)
        found = self.find_candidates(detections, regions, hands, changed, moving, person, size)

        if self.box is None:
            self.consider(found, now)
            return self.box

        match = max(found, key=lambda c: overlap(c.box, self.box), default=None)
        if match and overlap(match.box, self.box) >= 0.15 and not self.manual:
            self.box = follow_box(self.box, match.box)
            self.held, self.last_support = match.held, now
            self.points = None
        elif self.scene.differs_from_snapshot(self.box, size) >= MIN_ITEM_DIFFERENCE:
            self.last_support = now

        # A match can be a ghost of the item, so the box must really differ from the room.
        if self.scene.differs_from_snapshot(self.box, size) < MIN_ITEM_DIFFERENCE:
            self.last_support = min(self.last_support, now - LOST_SECONDS)

        # Someone walking by can look like a new item at first, until their outline fills the box.
        scale_x, scale_y = SCENE_SIZE[0] / width, SCENE_SIZE[1] / height
        on_body = share_inside(self.body, self.box, scale_x, scale_y) >= BODY_SHARE
        if on_body and not self.held and not self.manual and not self.named(detections):
            self.body_frames += 1
        else:
            self.body_frames = 0
        if self.body_frames >= BODY_FRAMES:
            self.last_support = min(self.last_support, now - LOST_SECONDS)

        if not self.manual:
            self.consider_switching(found, moving, size, now)

        outside = self.box[2] < 0 or self.box[0] > width or self.box[3] < 0 or self.box[1] > height
        if outside or now - self.last_support > LOST_SECONDS:
            self.clear_item()
        return self.box

    def named(self, detections: list[Detection]) -> bool:
        """Return whether the detector agrees there's an object in the box."""
        return any(
            d.name != "person" and d.score >= MIN_SCORE and overlap(d.box, self.box) >= 0.15
            for d in detections
        )

    def consider_switching(
        self,
        found: list[Candidate],
        moving: np.ndarray,
        frame_size: tuple[int, int],
        now: float,
    ) -> None:
        """Move to another item that's clearly being shown while this one sits still."""
        smallest = MIN_RIVAL_SIZE * box_area(self.box)
        others = [
            c
            for c in found
            if overlap(c.box, self.box) < 0.1 and (c.held or box_area(c.box) >= smallest)
        ]
        scale_x, scale_y = SCENE_SIZE[0] / frame_size[0], SCENE_SIZE[1] / frame_size[1]
        active = share_inside(moving, self.box, scale_x, scale_y) >= MIN_MOTION_SHARE
        if not others or (active and self.held):
            self.rival, self.rival_frames = None, 0
            return
        best = max(others, key=lambda c: (c.held, box_area(c.box)))
        if self.rival is not None and overlap(best.box, self.rival) >= 0.2:
            self.rival_frames += 1
        else:
            self.rival_frames = 1
        self.rival = best.box
        if self.rival_frames >= SWITCH_FRAMES:
            self.held = best.held
            self.scene.restore_snapshot()
            self.box = None
            self.lock(best.box, now)
            self.rival, self.rival_frames = None, 0

    def clear_item(self) -> None:
        self.scene.restore_snapshot()
        self.box, self.manual, self.held, self.points = None, False, False, None
        self.pending, self.pending_frames = None, 0

    def consider(self, found: list[Candidate], now: float) -> None:
        """Lock on once the same candidate has lasted for several frames."""
        if not found:
            self.pending, self.pending_frames = None, 0
            return
        best = max(found, key=lambda c: (c.held, c.source == "detector"))
        if self.pending is not None and overlap(best.box, self.pending) >= 0.2:
            self.pending_frames += 1
        else:
            self.pending_frames = 1
        self.pending = best.box
        if self.pending_frames >= LOCK_FRAMES:
            self.held = best.held
            self.lock(best.box, now)
