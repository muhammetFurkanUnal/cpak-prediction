"""
YOLO knee detector wrapper for the dual-knee inference path.

Loads the knee-detection ONNX model lazily on first use and exposes a single
function `detect_two_knees(img)` that returns the left and right knee bounding
boxes plus the recommended split-x coordinate that separates the two knees.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

KNEE_DETECTOR_PATH = (
    Path(__file__).parent.parent
    / "yolo-knee-detect" / "models" / "knee-detect-ep100.onnx"
)

_session: Optional[YOLO] = None


def _get_session() -> YOLO:
    global _session
    if _session is None:
        if not KNEE_DETECTOR_PATH.exists():
            raise FileNotFoundError(
                f"Knee detector model not found at {KNEE_DETECTOR_PATH}"
            )
        _session = YOLO(str(KNEE_DETECTOR_PATH), task="detect")
    return _session


def detect_two_knees(img_bgr: np.ndarray) -> dict:
    """
    Run knee detection on a BGR image and return the two knees plus the
    recommended vertical split coordinate.

    Returns
    -------
    {
      "left":    [x1, y1, x2, y2],   # leftmost knee bbox in original image coords
      "right":   [x1, y1, x2, y2],   # rightmost knee bbox
      "split_x": int,                # x where the image should be cut into halves
                                     # = midpoint of inner edges between the boxes
    }

    Raises
    ------
    ValueError
        If fewer than two knees are detected even after lowering the
        confidence threshold.
    """
    model = _get_session()

    # First pass at the default threshold; relax if we don't get >= 2 detections.
    boxes = None
    for conf in (0.25, 0.10, 0.05):
        res = model.predict(source=img_bgr, conf=conf, verbose=False)[0]
        if len(res.boxes) >= 2:
            boxes = res.boxes
            break

    if boxes is None or len(boxes) < 2:
        raise ValueError(
            "Could not detect two knees in the image (got "
            f"{0 if boxes is None else len(boxes)} detection(s))."
        )

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    # If more than 2 detections, keep the 2 most confident.
    if len(xyxy) > 2:
        top2 = np.argsort(-conf)[:2]
        xyxy = xyxy[top2]

    # Sort the two boxes left-to-right by x-center.
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    order = np.argsort(centers_x)
    left, right = xyxy[order[0]], xyxy[order[1]]

    # Split the image at the midpoint of the GAP between the two boxes
    # (right edge of left box, left edge of right box). This is more robust
    # than averaging centroids when the bboxes have different sizes.
    split_x = int(round((left[2] + right[0]) / 2))

    # Clamp: if boxes overlap horizontally (left.x2 > right.x1), fall back to
    # midpoint of centroids so we still produce a valid split.
    if split_x <= left[0] or split_x >= right[2]:
        split_x = int(round((centers_x[order[0]] + centers_x[order[1]]) / 2))

    return {
        "left":    [float(v) for v in left],
        "right":   [float(v) for v in right],
        "split_x": int(split_x),
    }
