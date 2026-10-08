"""Command line interface for the notebook experiments."""

import argparse

import cv2

from .detectors import BallDetector, HaarDetector, YoloDetector
from .tracking import MultiObjectTracker
from .video import run_video


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect objects and predict motion in video.")
    parser.add_argument("--detector", choices=("balls", "haar", "yolo"), default="balls")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Input video file")
    source.add_argument("--camera", type=int, help="Camera device index, for example 0")
    parser.add_argument("--output", help="Annotated .mp4 or .avi output")
    parser.add_argument("--no-display", action="store_true", help="Process without an OpenCV window")
    parser.add_argument("--detect-every", type=positive_int, default=1)
    parser.add_argument("--max-objects", type=positive_int, default=20)
    parser.add_argument("--max-distance", type=float, default=100.0, help="Association distance in pixels")
    parser.add_argument("--max-missed", type=int, default=30, help="Allowed unmatched detector frames")
    parser.add_argument("--cascade", help="Optional Haar XML path")
    parser.add_argument("--config", help="YOLOv3 Darknet .cfg path")
    parser.add_argument("--weights", help="YOLOv3 .weights path")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)
    if args.detector == "yolo" and (not args.config or not args.weights):
        parser.error("YOLO requires --config and --weights")
    try:
        tracker = MultiObjectTracker(args.max_objects, args.max_distance, args.max_missed)
        if args.detector == "balls":
            detector = BallDetector()
        elif args.detector == "haar":
            detector = HaarDetector(args.cascade)
        else:
            detector = YoloDetector(args.config, args.weights, args.confidence, args.nms_threshold)
        count = run_video(args.camera if args.camera is not None else args.video,
                          detector, tracker, args.output, not args.no_display, args.detect_every)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Processed {count} frames.")
    return 0
