# Motion Prediction

Object detection and motion prediction in video using OpenCV and independent Kalman filters. This project extracts the original ball and human tracking notebooks into reusable Python modules and a configurable command line application.

## Features

- Detect circular objects with Hough circles.
- Detect people with a bundled Haar full body cascade or external YOLOv3 files.
- Track up to a configurable number of objects with persistent numeric IDs.
- Predict positions on frames where detection is skipped.
- Read video files or cameras and optionally save annotated video.
- Run without a display using `--no-display`.

Red circles show current measurements. Yellow circles show the Kalman prior for the current frame, computed before its measurement is applied. These priors are one frame predictions based on earlier state, rather than long horizon trajectory forecasts.

## Install

Use Python 3.10 or newer:

```bash
git clone https://github.com/AdityaKharmaleGoat7/Motion-Prediction.git
cd Motion-Prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Dependencies are NumPy and OpenCV. Video files and YOLO model files are not included.

## Run

Detect balls in your video:

```bash
python -m motion_prediction --detector balls --video path/to/balls.mp4
```

Detect humans with the included Haar cascade:

```bash
python -m motion_prediction --detector haar --video path/to/people.mp4
```

Use YOLOv3 with a compatible COCO trained Darknet configuration and weights that you provide:

```bash
python -m motion_prediction --detector yolo --video path/to/crowd.mp4 --config models/yolov3.cfg --weights models/yolov3.weights
```

Predict between detections and save the result without opening a window:

```bash
python -m motion_prediction --detector haar --video path/to/people.mp4 --detect-every 5 --no-display --output outputs/tracked.mp4
```

Read a camera:

```bash
python -m motion_prediction --detector balls --camera 0
```

The installed `motion-prediction` command accepts the same options. Press `q` to stop an interactive run. Use `python -m motion_prediction --help` for all options. On Linux, interactive display requires a desktop session and OpenCV system libraries such as `libGL`. For servers, use `--no-display`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `motion_prediction/kalman.py` | Independent constant velocity filters |
| `motion_prediction/detectors.py` | Hough, Haar, and YOLOv3 detection |
| `motion_prediction/tracking.py` | Object association, IDs, and track expiry |
| `motion_prediction/video.py` | Frame processing, overlays, and video export |
| `motion_prediction/cli.py` | Command line arguments and application setup |
| `motion_prediction/assets/` | Bundled Haar model and its original notice |
| `notebooks/` | Original experiments retained for reference |
| `docs/` | Architecture, migration, and model setup |
| `tests/` | Tracking, detector decoding, and synthetic video tests |
| `coco.names` | Original COCO class names reference |

Read [the documentation](docs/README.md), [the notebook migration notes](notebooks/README.md), and [asset information](motion_prediction/assets/README.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

CI runs tests on Python 3.10 and 3.12. Tests generate short synthetic videos and require no downloaded weights, camera, or display.

## Limitations

This is a learning project for motion prediction in image coordinates. Hough detection finds circles, not specifically green balls. Haar detection is sensitive to pose and image quality. Greedy position association can switch identities when objects cross, overlap, or move farther than `--max-distance`. It does not use appearance embeddings or solve a global assignment problem. Predictions during long occlusions can drift. Velocity uses pixels per frame, with a fixed time step of one frame.

## License

Original project code is available under the [MIT License](LICENSE). The bundled Haar XML keeps its own copyright and redistribution notice. The included research PDF and external models are third party materials and are not relicensed under MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
