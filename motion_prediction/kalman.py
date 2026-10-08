"""Independent constant velocity Kalman filters in image coordinates."""

import cv2
import numpy as np


class KalmanFilter:
    """Track [x, y, vx, vy], with velocity measured in pixels per frame."""

    def __init__(self, center, dt=1.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[center[0]], [center[1]], [0], [0]], np.float32)

    def predict(self):
        """Advance one frame and return the prior position for that frame."""
        state = self.kf.predict()
        return float(state[0, 0]), float(state[1, 0])

    def correct(self, center):
        """Update the predicted state with a current frame measurement."""
        state = self.kf.correct(np.asarray(center, np.float32).reshape(2, 1))
        return float(state[0, 0]), float(state[1, 0])
