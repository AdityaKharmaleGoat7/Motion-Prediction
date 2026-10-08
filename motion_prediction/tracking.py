"""Bounded multi object tracking with gated nearest neighbor association."""

from dataclasses import dataclass
from math import dist

from .kalman import KalmanFilter


@dataclass
class Track:
    id: int
    filter: KalmanFilter
    predicted: tuple[float, float]
    measured: tuple[float, float] | None = None
    missed: int = 0


class MultiObjectTracker:
    def __init__(self, max_objects=20, max_distance=100.0, max_missed=30):
        if max_objects < 1 or max_distance <= 0 or max_missed < 0:
            raise ValueError("Invalid tracking limits")
        self.max_objects = max_objects
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 0

    def update(self, detections=None):
        """Advance every frame. None means detection was intentionally skipped.

        An empty list means detection ran and found no objects. Only detector
        frames count toward max_missed, so skipped frames do not delete tracks.
        """
        for track in self.tracks.values():
            track.predicted = track.filter.predict()
            track.measured = None
        if detections is None:
            return list(self.tracks.values())
        candidates = sorted(
            (dist(track.predicted, detection.center), track.id, index)
            for track in self.tracks.values()
            for index, detection in enumerate(detections)
        )
        used_tracks, used_detections = set(), set()
        for distance, track_id, index in candidates:
            if distance > self.max_distance:
                break
            if track_id in used_tracks or index in used_detections:
                continue
            track = self.tracks[track_id]
            track.measured = detections[index].center
            track.filter.correct(track.measured)
            track.missed = 0
            used_tracks.add(track_id)
            used_detections.add(index)
        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]
        for index, detection in enumerate(detections):
            if index in used_detections or len(self.tracks) >= self.max_objects:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(
                track_id, KalmanFilter(detection.center), detection.center, detection.center,
            )
        return list(self.tracks.values())
