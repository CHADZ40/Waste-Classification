"""Find and classify waste live from the camera in a desktop window.

MediaPipe sees objects, hands and people. The tracker uses them to follow the item being
shown and to ignore the room. The waste model classifies what's inside the box.

Click any object to follow it. Keys: 1-5 switch model, r reset, q or Esc quit.
"""

import argparse
import math
import ssl
import threading
import time
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import certifi
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
from PIL import Image, ImageDraw, ImageFont

from waste_classifier.model import load_checkpoint
from waste_classifier.predict import predict_image
from waste_classifier.tracking import SCENE_SIZE, Box, Detection, ItemTracker, overlap
from waste_classifier.utils import get_device

MODELS = {
    "Final model": "transfer_large.pt",
    "Fine-tuned Model B": "transfer_real_world.pt",
    "Model B": "transfer.pt",
    "Model A": "basic_cnn.pt",
    "Model A (first run)": "basic_cnn_strong_weights.pt",
}

MEDIAPIPE_URL = "https://storage.googleapis.com/mediapipe-models"
DETECTOR_FILE = "efficientdet_lite0.tflite"
DETECTOR_PATH = "object_detector/efficientdet_lite0/float16/latest"
HANDS_FILE = "hand_landmarker.task"
HANDS_PATH = "hand_landmarker/hand_landmarker/float16/latest"
PEOPLE_FILE = "selfie_multiclass_256x256.tflite"
PEOPLE_PATH = "image_segmenter/selfie_multiclass_256x256/float32/latest"
TOUCH_FILE = "magic_touch.tflite"
TOUCH_PATH = "interactive_segmenter/magic_touch/float32/latest"

WINDOW = "Waste Classifier"
NEON = (80, 255, 120)  # RGB
MUTED = (170, 178, 172)
INK = (12, 18, 14)
MIN_DETECTION_SCORE = 0.2
SAMPLE_SECONDS = 0.25
SLOW_VISION_EVERY_FRAMES = 3
FONT_FILES = ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc")


@dataclass
class Result:
    label: str
    confidence: float
    sure: bool


@dataclass
class Scan:
    """What the scanner currently knows, for the display."""

    box: Box | None = None
    result: Result | None = None


def find_models(models_dir: str | Path) -> dict[str, Path]:
    """Return the models that have a checkpoint on disk, best model first."""
    paths = {name: Path(models_dir) / filename for name, filename in MODELS.items()}
    available = {name: path for name, path in paths.items() if path.exists()}
    if not available:
        raise FileNotFoundError(f"No trained models found in {models_dir}")
    return available


def summarize(samples: list[dict[str, float]], min_confidence: float) -> Result | None:
    """Average several predictions, so one blurry frame can't change the answer."""
    if not samples:
        return None
    averaged = {name: float(np.mean([s[name] for s in samples])) for name in samples[0]}
    label = max(averaged, key=averaged.get)
    return Result(label, averaged[label], averaged[label] >= min_confidence)


def square_crop(frame: np.ndarray, box: Box, margin: float = 0.15) -> np.ndarray:
    """Cut a square around a box, because the waste models expect a centered item."""
    height, width = frame.shape[:2]
    side = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin)
    side = min(side, width, height)
    x0 = min(max((box[0] + box[2] - side) / 2, 0), width - side)
    y0 = min(max((box[1] + box[3] - side) / 2, 0), height - side)
    return frame[int(y0) : int(y0 + side), int(x0) : int(x0 + side)]


class BoxFilter:
    """One Euro filter: steady when an item is still, and quick when it moves."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.015, speed_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.speed_cutoff = min_cutoff, beta, speed_cutoff
        self.reset()

    def reset(self) -> None:
        self.value = self.speed = self.time = None

    @staticmethod
    def blend(cutoff, elapsed: float):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * elapsed))

    def __call__(self, box: Box, now: float) -> Box:
        target = np.array(box, dtype=np.float64)
        if self.value is None or overlap(tuple(self.value), box) < 0.05:
            self.value, self.speed, self.time = target, np.zeros(4), now  # a new item: jump to it
            return box
        elapsed = max(now - self.time, 1e-3)
        speed = (target - self.value) / elapsed
        self.speed += self.blend(self.speed_cutoff, elapsed) * (speed - self.speed)
        cutoff = self.min_cutoff + self.beta * np.abs(self.speed)
        self.value = self.value + self.blend(cutoff, elapsed) * (target - self.value)
        self.time = now
        return tuple(self.value)


def download_model(folder: Path, url_path: str, filename: str) -> Path:
    """Fetch a MediaPipe model once. It's stored with the other models, outside git."""
    path = folder / filename
    if not path.exists():
        print(f"Downloading {filename}...")
        folder.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(f"{MEDIAPIPE_URL}/{url_path}/{filename}", context=context) as r:
            path.write_bytes(r.read())
    return path


class Vision:
    """MediaPipe object and hand detection on RGB frames."""

    def __init__(self, models_dir: str | Path):
        folder = Path(models_dir) / "mediapipe"
        self.objects = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(download_model(folder, DETECTOR_PATH, DETECTOR_FILE))
                ),
                running_mode=vision.RunningMode.IMAGE,
                max_results=8,
                score_threshold=MIN_DETECTION_SCORE,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(download_model(folder, HANDS_PATH, HANDS_FILE))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self.people = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(download_model(folder, PEOPLE_PATH, PEOPLE_FILE))
                ),
                output_category_mask=True,
                output_confidence_masks=False,
            )
        )
        self.touch = vision.InteractiveSegmenterLegacy.create_from_options(
            vision.InteractiveSegmenterLegacyOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(download_model(folder, TOUCH_PATH, TOUCH_FILE))
                ),
                output_category_mask=True,
                output_confidence_masks=False,
            )
        )

    def detect_objects(self, rgb: np.ndarray) -> list[Detection]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        found = []
        for item in self.objects.detect(image).detections:
            box, category = item.bounding_box, item.categories[0]
            corners = (
                box.origin_x,
                box.origin_y,
                box.origin_x + box.width,
                box.origin_y + box.height,
            )
            found.append(Detection(corners, category.score, category.category_name))
        return found

    def detect_hands(self, rgb: np.ndarray, timestamp_ms: int) -> list[Box]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        height, width = rgb.shape[:2]
        boxes = []
        for hand in self.hands.detect_for_video(image, timestamp_ms).hand_landmarks:
            xs, ys = [p.x * width for p in hand], [p.y * height for p in hand]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes


    def segment_people(self, rgb: np.ndarray) -> np.ndarray:
        """Return a small mask of people: hair, skin and clothes, but not accessories."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        classes = self.people.segment(image).category_mask.numpy_view().squeeze()
        mask = ((classes >= 1) & (classes <= 4)).astype(np.uint8) * 255
        return cv2.resize(mask, SCENE_SIZE, interpolation=cv2.INTER_NEAREST)

    def segment_at(self, rgb: np.ndarray, x: int, y: int) -> Box | None:
        """Return the box of the object at a point, or None if nothing was found there."""
        height, width = rgb.shape[:2]
        region = vision.InteractiveSegmenterLegacyRegionOfInterest
        point = region(
            format=region.Format.KEYPOINT, keypoint=NormalizedKeypoint(x / width, y / height)
        )
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        classes = self.touch.segment(image, point).category_mask.numpy_view().squeeze()
        mask = (classes > 0).astype(np.uint8)
        if mask.mean() > 0.6:
            mask = 1 - mask  # the model's foreground value isn't fixed
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        label = labels[y, x]
        if label == 0:
            return None
        left, top, box_width, box_height, _ = stats[label]
        return (float(left), float(top), float(left + box_width), float(top + box_height))


class Scanner(threading.Thread):
    """Runs detection and classification in the background, so the video stays smooth."""

    def __init__(self, seeing, models, model_name, device, interval, min_confidence):
        super().__init__(daemon=True)
        self.seeing, self.models, self.model_name = seeing, models, model_name
        self.device, self.interval, self.min_confidence = device, interval, min_confidence
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.scan = Scan()
        self.samples = []
        self.tracker = ItemTracker()
        self.hands, self.frames = [], 0
        self.people = np.zeros(SCENE_SIZE[::-1], np.uint8)
        self.click_point = None
        self.want_reset = False
        self.item_number = 0
        self.started = self.last_update = time.monotonic()
        self.last_sample = 0.0

    def submit(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frame = frame

    def switch_model(self, name: str) -> None:
        with self.lock:
            self.model_name = name
            self.samples, self.scan.result = [], None
            self.last_update = time.monotonic()

    def click(self, x: int, y: int) -> None:
        with self.lock:
            self.click_point = (x, y)

    def reset(self) -> None:
        with self.lock:
            self.want_reset = True

    def state(self) -> Scan:
        with self.lock:
            return Scan(self.scan.box, self.scan.result)

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
                name = self.model_name
                point, self.click_point = self.click_point, None
                reset, self.want_reset = self.want_reset, False
            if frame is None:
                time.sleep(0.005)
                continue
            self.process(frame, name, point, reset)

    def process(
        self, frame: np.ndarray, name: str, point: tuple[int, int] | None, reset: bool
    ) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.monotonic()
        if reset:
            self.tracker.reset()

        found = self.seeing.detect_objects(rgb)
        self.frames += 1
        if self.frames % SLOW_VISION_EVERY_FRAMES == 1:
            elapsed_ms = int((now - self.started) * 1000)
            self.hands = self.seeing.detect_hands(rgb, elapsed_ms)
            self.people = self.seeing.segment_people(rgb)

        if point is not None:
            chosen = self.seeing.segment_at(rgb, *point)
            if chosen is not None:
                self.tracker.choose(chosen, now)

        box = self.tracker.update(frame, now, found, self.hands, self.people)

        scores = None
        if box is not None and now - self.last_sample >= SAMPLE_SECONDS:
            model, meta = self.models[name]
            crop = square_crop(rgb, box)
            if crop.size:
                scores = predict_image(model, meta, Image.fromarray(crop), self.device)
                self.last_sample = now

        with self.lock:
            if name != self.model_name:
                return
            if box is None or self.tracker.item_number != self.item_number:
                # A different item must not inherit the last item's answer.
                self.item_number = self.tracker.item_number
                self.samples, self.scan.result = [], None
                self.last_update = now
            self.scan.box = box
            if scores:
                self.samples.append(scores)
            if now - self.last_update >= self.interval:
                self.scan.result = summarize(self.samples, self.min_confidence) or self.scan.result
                self.samples, self.last_update = [], now


@cache
def load_font(size: int, weight: str = "Regular") -> ImageFont.ImageFont:
    for path in FONT_FILES:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            try:
                font.set_variation_by_name(weight)
            except OSError:
                pass  # not a variable font
            return font
    return ImageFont.load_default(size)


@cache
def rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Return a smooth rounded-rectangle mask from 0 to 1, drawn large and scaled down."""
    mask = Image.new("L", (width * 4, height * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, width * 4 - 1, height * 4 - 1], radius=radius * 4, fill=255
    )
    small = mask.resize((width, height), Image.LANCZOS)
    return np.asarray(small, dtype=np.float32)[:, :, None] / 255


def chip_size(text: str, font: ImageFont.ImageFont, scale: float) -> tuple[int, int]:
    return int(font.getlength(text) + 18 * scale), int(24 * scale)


def draw_chip(
    frame: np.ndarray,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    scale: float,
    fill: tuple[int, int, int],
    ink: tuple[int, int, int],
    opacity: float = 1.0,
) -> None:
    """Draw a small rounded text label with its top-left corner at x, y."""
    width, height = chip_size(text, font, scale)
    chip = Image.new("RGB", (width, height), fill)
    ImageDraw.Draw(chip).text((width / 2, height / 2), text, font=font, fill=ink, anchor="mm")
    mask = rounded_mask(width, height, int(7 * scale)) * opacity
    pixels = np.asarray(chip, dtype=np.float32)[:, :, ::-1]
    region = frame[y : y + height, x : x + width].astype(np.float32)
    frame[y : y + height, x : x + width] = (region * (1 - mask) + pixels * mask).astype(np.uint8)


def draw_tracking_box(frame: np.ndarray, box: Box, scale: float) -> None:
    """Draw a thin neon box with short corners and a faint glow."""
    height, width = frame.shape[:2]
    x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
    x1, y1 = min(int(box[2]), width - 1), min(int(box[3]), height - 1)
    color = NEON[::-1]
    arm = max(int(min(x1 - x0, y1 - y0) * 0.1), 6)
    corner = max(int(2 * scale), 2)

    def outline(image: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> None:
        a, b, c, d = x0 - offset_x, y0 - offset_y, x1 - offset_x, y1 - offset_y
        cv2.rectangle(image, (a, b), (c, d), color, 1, cv2.LINE_AA)
        for px, dx in ((a, 1), (c, -1)):
            for py, dy in ((b, 1), (d, -1)):
                cv2.line(image, (px, py), (px + dx * arm, py), color, corner, cv2.LINE_AA)
                cv2.line(image, (px, py), (px, py + dy * arm), color, corner, cv2.LINE_AA)

    pad = int(14 * scale)
    gx0, gy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    gx1, gy1 = min(x1 + pad, width), min(y1 + pad, height)
    glow = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
    outline(glow, gx0, gy0)
    glow = cv2.GaussianBlur(glow, (0, 0), 4 * scale)
    frame[gy0:gy1, gx0:gx1] = cv2.addWeighted(frame[gy0:gy1, gx0:gx1], 1, glow, 0.7, 0)
    outline(frame)


def draw_box_label(frame: np.ndarray, scan: Scan, scale: float) -> None:
    """Put the result on the box's top edge, or just inside it near the top of the frame."""
    result = scan.result
    if result and result.sure:
        text, fill, ink = f"{result.label}  {result.confidence:.0%}", NEON, INK
    elif result:
        text, fill, ink = "Not sure", INK, MUTED
    else:
        text, fill, ink = "Scanning", INK, NEON

    font = load_font(int(13 * scale), "Semibold")
    label_width, label_height = chip_size(text, font, scale)
    gap = int(5 * scale)
    height, width = frame.shape[:2]
    x = min(max(int(scan.box[0]), 0), width - label_width)
    y = int(scan.box[1]) - label_height - gap
    if y < 0:
        y = max(int(scan.box[1]), 0) + gap
    y = min(y, height - label_height)
    draw_chip(frame, x, y, text, font, scale, fill, ink, opacity=0.94)


def draw_overlay(frame: np.ndarray, scan: Scan, model_name: str) -> np.ndarray:
    """Return a BGR frame with the tracking box, its label and a small model tag."""
    drawn = frame.copy()
    scale = frame.shape[0] / 720
    if scan.box:
        draw_tracking_box(drawn, scan.box, scale)
        draw_box_label(drawn, scan, scale)
    margin = int(14 * scale)
    tag = f"{model_name}  ·  1–5 model  ·  Click to pick  ·  R reset  ·  Q quit"
    draw_chip(
        drawn, margin, margin, tag, load_font(int(12 * scale)), scale, INK, MUTED, opacity=0.6
    )
    return drawn


def open_camera(source: str) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(int(source) if source.isdigit() else source)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the camera. Allow your terminal app in System Settings > "
            "Privacy & Security > Camera, then restart it."
        )
    return camera


def run(args: argparse.Namespace) -> None:
    device = get_device()
    available = find_models(args.models_dir)
    models = {name: load_checkpoint(path, device) for name, path in available.items()}
    names = list(models)
    current = names[0]

    camera = open_camera(args.source)
    scanner = Scanner(
        Vision(args.models_dir), models, current, device, args.interval, args.min_confidence
    )
    scanner.start()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            scanner.click(x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)
    box_filter = BoxFilter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped sending video.")
            frame = cv2.flip(frame, 1)  # mirror, so moving an item feels natural
            scanner.submit(frame)

            scan = scanner.state()
            if scan.box:
                scan.box = box_filter(scan.box, time.monotonic())
            else:
                box_filter.reset()
            cv2.imshow(WINDOW, draw_overlay(frame, scan, current))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                scanner.reset()
            if ord("1") <= key < ord("1") + len(names):
                current = names[key - ord("1")]
                scanner.switch_model(current)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        scanner.stop()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera number, or a video file")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--interval", type=float, default=1.5, help="seconds between results")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
