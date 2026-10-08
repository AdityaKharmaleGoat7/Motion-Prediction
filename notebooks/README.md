# Original notebooks

These notebooks preserve the original experiments for reference. Their code has been consolidated into the Python application. Use the application for normal runs.

| Notebook or cell | Python equivalent |
| --- | --- |
| `BallsMotionPrediction.ipynb` | `BallDetector`, `KalmanFilter`, and the balls CLI mode |
| `main.ipynb`, cell 0 | `HaarDetector` and the haar CLI mode |
| `main.ipynb`, cell 1 | `YoloDetector` and the yolo CLI mode |
| `main.ipynb`, cell 2 | Shared tracker using current measurements and persistent IDs |
| `main.ipynb`, cell 3 | YOLO mode with `--detect-every 30` |

The original notebooks reference local videos named `balls.mp4`, `green.mp4`, `crowd.mp4`, and `fast.mp4`, plus external YOLO files. These videos and models are not in the repository. The Haar XML has moved into `motion_prediction/assets/`; update the notebook's relative path if you run the legacy Haar cell.

The Python implementation fixes shared filter state in the ball experiment, the maximum object boundary check, detection order association, delayed measurement updates, and stale measurements on skipped frames. The original cells remain unchanged so that the migration can be compared with the source experiments.

For optional notebook exploration:

```bash
python -m pip install -e '.[notebooks]'
jupyter lab notebooks/
```
