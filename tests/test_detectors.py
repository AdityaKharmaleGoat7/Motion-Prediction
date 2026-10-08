import cv2
import numpy as np
import pytest

from motion_prediction.detectors import BallDetector, HaarDetector, YoloDetector


def test_blank_frames_have_no_balls_or_humans():
    frame = np.zeros((240, 320, 3), np.uint8)
    assert BallDetector().detect(frame) == []
    assert HaarDetector().detect(frame) == []


def test_missing_model_assets_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Haar"):
        HaarDetector(tmp_path / "missing.xml")
    with pytest.raises(FileNotFoundError, match="YOLO"):
        YoloDetector(tmp_path / "missing.cfg", tmp_path / "missing.weights")


def test_yolo_objectness_nms_and_non_person_filtering():
    detector = YoloDetector.__new__(YoloDetector)
    detector.confidence, detector.nms_threshold = 0.5, 0.4
    # x, y, w, h, objectness, person probability, other class probability
    rows = np.array([
        [.5, .5, .4, .4, .9, .9, .1],
        [.5, .5, .4, .4, .8, .9, .1],
        [.2, .2, .1, .1, .1, .99, .01],
        [.8, .8, .1, .1, .9, .1, .9],
    ], np.float32)
    detections = detector.decode([rows], (100, 100, 3))
    assert len(detections) == 1
    assert detections[0].center == pytest.approx((50, 50), abs=1)
    assert detections[0].confidence == pytest.approx(.81)
    assert detector.decode([np.empty((0, 7))], (100, 100, 3)) == []


def test_hough_finds_synthetic_circle():
    frame = np.zeros((180, 180, 3), np.uint8)
    cv2.circle(frame, (90, 90), 25, (255, 255, 255), 3)
    circles = BallDetector().detect(frame)
    assert circles
    assert circles[0].center == pytest.approx((90, 90), abs=4)
