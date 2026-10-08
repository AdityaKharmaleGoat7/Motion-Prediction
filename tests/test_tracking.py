import numpy as np
import pytest

from motion_prediction.detectors import Detection
from motion_prediction.kalman import KalmanFilter
from motion_prediction.tracking import MultiObjectTracker


def detection(x, y=20):
    return Detection((x, y), (int(x) - 5, int(y) - 5, 10, 10))


def test_filters_do_not_share_state():
    a, b = KalmanFilter((10, 20)), KalmanFilter((200, 300))
    assert a.kf is not b.kf
    for x in range(15, 60, 5):
        a.predict()
        a.correct((x, 20))
    assert b.predict() == pytest.approx((200, 300))
    assert a.predict()[0] > 50


def test_track_ids_survive_detection_reordering():
    tracker = MultiObjectTracker(max_distance=30)
    tracker.update([detection(10), detection(200)])
    tracks = tracker.update([detection(205), detection(15)])
    assert {t.id: t.measured for t in tracks} == {0: (15, 20), 1: (205, 20)}


def test_cap_and_expiry():
    tracker = MultiObjectTracker(max_objects=2, max_missed=1)
    assert len(tracker.update([detection(10), detection(100), detection(200)])) == 2
    assert len(tracker.update([])) == 2
    assert tracker.update([]) == []


def test_skipped_detection_predicts_without_reusing_measurements():
    tracker = MultiObjectTracker(max_missed=0)
    tracker.update([detection(10)])
    tracker.update([detection(20)])
    last_prior = tracker.tracks[0].predicted[0]
    tracks = tracker.update(None)
    assert len(tracks) == 1
    assert tracks[0].measured is None
    assert tracks[0].missed == 0
    assert tracks[0].predicted[0] > last_prior
    assert tracker.update([]) == []


def test_distance_gate_prevents_unrelated_match():
    tracker = MultiObjectTracker(max_distance=10)
    tracker.update([detection(10)])
    tracks = tracker.update([detection(200)])
    assert tracks[0].measured is None
    assert tracks[1].measured == (200, 20)


@pytest.mark.parametrize("kwargs", [{"max_objects": 0}, {"max_distance": 0}, {"max_missed": -1}])
def test_invalid_tracking_limits(kwargs):
    with pytest.raises(ValueError):
        MultiObjectTracker(**kwargs)


def test_linear_motion_prediction():
    tracker = MultiObjectTracker()
    for frame in range(40):
        tracks = tracker.update([detection(100 + frame * 3)])
    assert tracks[0].predicted[0] == pytest.approx(217, abs=1)
    assert np.isfinite(tracker.tracks[0].filter.kf.errorCovPost).all()
