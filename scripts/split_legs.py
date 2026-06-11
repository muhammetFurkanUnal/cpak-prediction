"""
Split full-leg (long-leg) radiographs that contain both legs into two
single-leg images, using the YOLO knee-detection model to find where to cut.

For each input image the knee detector locates the two knees and returns the
vertical split coordinate (the midpoint of the gap between the two knees). The
image is then cut vertically at that coordinate into a left half and a right
half (full height preserved), producing one single-leg image per side.

Output files keep the original stem with a side suffix:
    1234.png  ->  1234.l.png  (left half of the image)
                  1234.r.png  (right half of the image)

Usage:
    python split_legs.py <input> [--out DIR] [--model PATH] [--conf 0.25]

Args:
    input:    Path to a single image or a directory of images (searched
              recursively).
    --out:    Output directory. Default: ./split_out
    --model:  Path to the YOLO knee-detector weights (.onnx or .pt).
              Default: ../yolo-knee-detect/models/knee-detect-ep100.onnx
    --conf:   Starting confidence threshold (relaxed automatically if fewer
              than two knees are found). Default: 0.25
    --fallback-center:
              If the detector cannot find two knees, split the image down the
              middle (50%) instead of skipping it.

Examples:
    python split_legs.py ./llr_images --out ./single_legs
    python split_legs.py 1234.png --fallback-center
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "yolo-knee-detect" / "models" / "knee-detect-ep100.onnx"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_model: Optional[YOLO] = None


def get_model(model_path: Path) -> YOLO:
    global _model
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        _model = YOLO(str(model_path), task="detect")
    return _model


def find_split_x(model: YOLO, img_bgr: np.ndarray, start_conf: float) -> Optional[int]:
    """Return the x coordinate to cut the image into two single-leg halves,
    or None if two knees could not be detected."""
    boxes = None
    for conf in (start_conf, 0.10, 0.05):
        res = model.predict(source=img_bgr, conf=conf, verbose=False)[0]
        if len(res.boxes) >= 2:
            boxes = res.boxes
            break

    if boxes is None or len(boxes) < 2:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    # Keep the two most confident detections if there are extras.
    if len(xyxy) > 2:
        xyxy = xyxy[np.argsort(-conf)[:2]]

    # Sort left-to-right by x-center.
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    order = np.argsort(centers_x)
    left, right = xyxy[order[0]], xyxy[order[1]]

    # Cut at the midpoint of the gap between the two knees.
    split_x = int(round((left[2] + right[0]) / 2))

    # Fall back to centroid midpoint if the boxes overlap horizontally.
    if split_x <= left[0] or split_x >= right[2]:
        split_x = int(round((centers_x[order[0]] + centers_x[order[1]]) / 2))

    return split_x


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--out", default="./split_out",
                        help="Output directory (default: ./split_out)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"Knee-detector weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Starting confidence threshold (default: 0.25)")
    parser.add_argument("--fallback-center", action="store_true",
                        help="Split at 50%% width when no two knees are detected")
    args = parser.parse_args()

    images = collect_images(Path(args.input))
    if not images:
        print("No images found.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(Path(args.model))
    print(f"Running on {len(images)} image(s)... output -> {out_dir}")

    for src in images:
        img = cv2.imread(str(src))
        if img is None:
            print(f"  skip (decode failed): {src.name}")
            continue

        width = img.shape[1]
        split_x = find_split_x(model, img, args.conf)

        if split_x is None:
            if args.fallback_center:
                split_x = width // 2
                print(f"  {src.name}: no two knees detected -> center split")
            else:
                print(f"  skip (no two knees detected): {src.name}")
                continue

        # Clamp so both halves are non-empty.
        split_x = max(1, min(split_x, width - 1))

        left_part = img[:, :split_x]
        right_part = img[:, split_x:]

        left_path = out_dir / f"{src.stem}.l{src.suffix}"
        right_path = out_dir / f"{src.stem}.r{src.suffix}"
        cv2.imwrite(str(left_path), left_part)
        cv2.imwrite(str(right_path), right_part)
        print(f"  {src.name}: split at x={split_x} -> {left_path.name}, {right_path.name}")


if __name__ == "__main__":
    main()
