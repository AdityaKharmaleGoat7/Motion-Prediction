"""Video processing and optional annotation export."""

from pathlib import Path

import cv2


def annotate(frame, tracks):
    for track in tracks:
        predicted = tuple(round(value) for value in track.predicted)
        cv2.circle(frame, predicted, 12, (0, 255, 255), 2)
        cv2.putText(frame, f"ID {track.id} prior", (predicted[0] + 15, predicted[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if track.measured is not None:
            measured = tuple(round(value) for value in track.measured)
            cv2.circle(frame, measured, 10, (0, 0, 255), 2)
            cv2.putText(frame, "Measured", (measured[0] + 15, measured[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def run_video(source, detector, tracker, output=None, display=True, detect_every=1):
    """Return processed frame count and always release OpenCV resources."""
    if detect_every < 1:
        raise ValueError("detect_every must be positive")
    if output and not isinstance(source, int) and Path(source).resolve() == Path(output).resolve():
        raise ValueError("Output must be different from the input video")
    capture = cv2.VideoCapture(source)
    writer = None
    frame_count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            fps = 30.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame) if frame_count % detect_every == 0 else None
            annotate(frame, tracker.update(detections))
            if output:
                if writer is None:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    codec = "MJPG" if Path(output).suffix.lower() == ".avi" else "mp4v"
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
                    if not writer.isOpened():
                        raise ValueError(f"Cannot create output video: {output}")
                writer.write(frame)
            frame_count += 1
            if display:
                cv2.imshow("Motion Prediction", frame)
                if cv2.waitKey(max(1, round(1000 / fps))) & 0xFF == ord("q"):
                    break
        if frame_count == 0:
            raise ValueError(f"Video source contains no readable frames: {source}")
        return frame_count
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
