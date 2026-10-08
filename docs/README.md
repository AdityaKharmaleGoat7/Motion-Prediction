# Project documentation

## Processing flow

Each video frame follows these steps:

1. Run the selected detector when the frame index is divisible by `detect_every`, beginning with frame zero.
2. Advance every existing Kalman filter by one frame.
3. On detector frames, pair measurements and predicted centers using the shortest available distance within `max_distance`.
4. Correct matched filters with current measurements, expire unmatched tracks, and create tracks for unmatched detections until `max_objects` is reached.
5. Draw measurements and current frame priors, then display or write the frame.

A skipped detector frame passes `None` into the tracker. This advances predictions without repeating stale measurements. An empty detection list means the detector ran and found nothing. Track expiry counts unmatched detector frames, not all video frames.

## State and association

The state is `[x, y, vx, vy]`. The transition matrix advances image position by velocity with a time step of one frame. Each track owns its filter, covariance matrices, and state. New tracks begin at their detected center with zero velocity.

Association sorts all track and detection distances, then greedily accepts disjoint pairs within the distance gate. IDs are monotonically increasing and are not reused after expiry. The gate defaults to 100 pixels and should be tuned to the resolution, speed, and detection interval. After 30 unmatched detector frames a track is retained; it expires on the next unmatched detector frame.

## Detector setup

### Balls

The Hough detector uses grayscale blur, radii from 1 to 40 pixels, and a minimum center distance of 20 pixels, matching the original experiment. It does not segment by color. To tune these settings for a different video, edit `BallDetector.detect` in `motion_prediction/detectors.py`.

### Haar

The default model is packaged at `motion_prediction/assets/haarcascade_fullbody.xml`. Use `--cascade path/to/model.xml` to supply another compatible model. The package resolves its bundled model independently of your working directory.

### YOLOv3

Supply `--config` and `--weights` for matching YOLOv3 Darknet files trained on COCO. Obtain them from the model publisher and follow the publisher's terms. Model weights are deliberately excluded from version control. The detector uses class index zero for people, so arbitrary custom class orders are not supported.

Confidence is objectness multiplied by the selected class probability. Boxes are clipped to the frame, and overlapping person boxes are suppressed using OpenCV NMS. Defaults are confidence 0.5 and NMS threshold 0.4. Override them with `--confidence` and `--nms-threshold`.

## Errors and resources

Missing models, unreadable input videos, and output writer failures cause a nonzero exit with an error message. Video output cannot overwrite the input path. Capture, writer, and display resources are released even if frame processing fails. MP4 uses the `mp4v` codec and AVI uses `MJPG`; codec availability depends on the OpenCV build.

## Validation

The test suite covers independent filter state, linear motion, reordered detections, capacity limits, distance gating, missing detections, skipped detection frames, YOLO decoding, synthetic circle detection, CLI errors, and synthetic video export. Real world accuracy and YOLO inference still require your video and model assets.
