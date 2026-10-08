"""Ball, Haar full body, and YOLOv3 person detectors from the notebooks."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    center: tuple[float, float]
    box: tuple[int, int, int, int]
    confidence: float = 1.0


class BallDetector:
    """Detect circular shapes using Hough circles, without a color restriction."""

    def detect(self, frame):
        gray = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=1, maxRadius=40,
        )
        if circles is None:
            return []
        return [
            Detection((float(x), float(y)), (x - r, y - r, 2 * r, 2 * r))
            for x, y, r in np.round(circles[0]).astype(int)
        ]


class HaarDetector:
    def __init__(self, cascade=None):
        path = Path(cascade) if cascade else Path(__file__).parent / "assets" / "haarcascade_fullbody.xml"
        if not path.is_file():
            raise FileNotFoundError(f"Haar cascade does not exist: {path}")
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise ValueError(f"Cannot load Haar cascade: {path}")

    def detect(self, frame):
        boxes = self.cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
        return [Detection((x + w / 2, y + h / 2), (x, y, w, h)) for x, y, w, h in boxes]


class YoloDetector:
    """Load an external Darknet YOLOv3 model and keep COCO person detections."""

    def __init__(self, config, weights, confidence=0.5, nms_threshold=0.4):
        for path in (config, weights):
            if not Path(path).is_file():
                raise FileNotFoundError(f"YOLO model file does not exist: {path}")
        if not 0 < confidence < 1 or not 0 < nms_threshold < 1:
            raise ValueError("YOLO thresholds must be between 0 and 1")
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNetFromDarknet(str(config), str(weights))
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.decode(self.net.forward(self.output_layers), frame.shape)

    def decode(self, outputs, shape):
        """Combine objectness and class probability, then suppress overlaps."""
        height, width = shape[:2]
        boxes, confidences = [], []
        for output in outputs:
            for row in output:
                scores = row[5:]
                class_id = int(np.argmax(scores))
                confidence = float(row[4] * scores[class_id])
                if class_id != 0 or confidence <= self.confidence:
                    continue
                cx, cy, w, h = row[:4] * np.array([width, height, width, height])
                left, top = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
                right, bottom = min(width, int(cx + w / 2)), min(height, int(cy + h / 2))
                if right <= left or bottom <= top:
                    continue
                boxes.append([left, top, right - left, bottom - top])
                confidences.append(confidence)
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.nms_threshold)
        return [
            Detection((x + w / 2, y + h / 2), (x, y, w, h), confidences[int(index)])
            for index in np.asarray(indices, dtype=int).reshape(-1)
            for x, y, w, h in [boxes[int(index)]]
        ]
