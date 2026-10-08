import cv2
import numpy as np
import pytest

from motion_prediction.cli import main
from motion_prediction.tracking import MultiObjectTracker
from motion_prediction.video import run_video


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_video_export_and_detection_schedule(tmp_path):
    source, output = tmp_path / "source.avi", tmp_path / "output.avi"
    make_video(source)
    detector = CountingDetector()
    assert run_video(str(source), detector, MultiObjectTracker(), str(output), False, 2) == 6
    assert detector.calls == 3
    capture = cv2.VideoCapture(str(output))
    try:
        count = 0
        while capture.read()[0]:
            count += 1
        assert count == 6
    finally:
        capture.release()


def test_input_cannot_be_overwritten(tmp_path):
    source = tmp_path / "source.avi"
    make_video(source)
    with pytest.raises(ValueError, match="different"):
        run_video(str(source), CountingDetector(), MultiObjectTracker(), str(source), False)


def test_missing_video_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot open"):
        run_video(str(tmp_path / "missing.mp4"), CountingDetector(), MultiObjectTracker(), display=False)


def test_cli_requires_yolo_assets():
    with pytest.raises(SystemExit) as error:
        main(["--detector", "yolo", "--video", "missing.mp4"])
    assert error.value.code == 2


def test_cli_processes_a_video(tmp_path, capsys):
    source = tmp_path / "source.avi"
    make_video(source)
    assert main(["--video", str(source), "--no-display"]) == 0
    assert "Processed 6 frames" in capsys.readouterr().out
