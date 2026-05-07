"""
Run the trained knee-detection ONNX model on one or more images and save
copies with bounding boxes drawn on top.

Usage:
    python infer.py <input> [--model PATH] [--out DIR] [--conf 0.25] [--iou 0.5]

Args:
    input:    Path to a single image or a directory of images.
    --model:  Path to the ONNX (or .pt) weights.
              Default: ../notebooks/models/knee-detect-ep100/last.onnx
    --out:    Output directory for annotated images. Default: ./infer_out
    --conf:   Confidence threshold. Default: 0.25
    --iou:    NMS IoU threshold. Default: 0.5

Examples:
    python infer.py ./train/images/0.png
    python infer.py ./test_imgs --conf 0.3 --out ./preds
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "notebooks" / "models" / "knee-detect-ep100" / "last.onnx"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def annotate(image, boxes, names) -> None:
    # boxes: ultralytics Boxes object (xyxy in original-image coordinates)
    for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(),
                               boxes.conf.cpu().numpy(),
                               boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"Path to weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--out", default="./out",
                        help="Directory to write annotated images (default: ./out)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold (default: 0.5)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    images = collect_images(Path(args.input))
    if not images:
        print("No images found.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    print(f"Running on {len(images)} image(s)... output -> {out_dir}")

    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    for src, res in zip(images, results):
        img = cv2.imread(str(src))
        if img is None:
            print(f"  skip (decode failed): {src}")
            continue
        annotate(img, res.boxes, res.names)
        save_path = out_dir / src.name
        cv2.imwrite(str(save_path), img)
        n = len(res.boxes)
        print(f"  {src.name}: {n} detection(s) -> {save_path}")


if __name__ == "__main__":
    main()
