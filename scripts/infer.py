#!/usr/bin/env python3
"""
CLI inference tool — single image → JSON + annotated JPEG.

Usage
-----
python3 scripts/infer.py <image_path> [--model MODEL] [--url URL] [--out-dir DIR]

Examples
--------
python3 scripts/infer.py /abs/path/to/image.jpg
python3 scripts/infer.py /abs/path/to/image.jpg --model sh14-ep1000
python3 scripts/infer.py /abs/path/to/image.jpg --out-dir /tmp/cpak_out

Output (saved to --out-dir, default: same folder as the image)
------
<stem>_result.json    — keypoints + metrics
<stem>_visualize.jpg  — mechanical axes drawn on image
<stem>_landmarks.jpg  — raw landmark points drawn on image
"""

import argparse
import json
import sys
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Run cpak inference on a single image.")
    p.add_argument("image", help="Absolute path to the input image (JPEG/PNG)")
    p.add_argument("--model", default=None,  help="Model name (default: first available)")
    p.add_argument("--url",   default="http://localhost:8000", help="API base URL")
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as image)")
    return p.parse_args()


def pick_model(base_url: str, requested: str | None) -> str:
    resp = requests.get(f"{base_url}/models", timeout=5)
    resp.raise_for_status()
    available = resp.json()["models"]
    if not available:
        sys.exit("No models available on the server.")
    if requested is None:
        return available[0]
    if requested not in available:
        sys.exit(f"Model '{requested}' not found. Available: {available}")
    return requested


def post_image(url: str, image_path: Path, as_bytes: bytes):
    return requests.post(
        url,
        files={"image": (image_path.name, as_bytes, "image/jpeg")},
        timeout=60,
    )


def main():
    args = parse_args()
    image_path = Path(args.image).resolve()

    if not image_path.exists():
        sys.exit(f"File not found: {image_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    base_url = args.url.rstrip("/")

    # Resolve model
    try:
        model = pick_model(base_url, args.model)
    except requests.exceptions.ConnectionError:
        sys.exit(f"Cannot reach server at {base_url}. Is it running?")

    print(f"Image  : {image_path}")
    print(f"Model  : {model}")
    print(f"Server : {base_url}")
    print(f"Out    : {out_dir}")
    print()

    image_bytes = image_path.read_bytes()

    # ── 1. JSON inference ──────────────────────────────────────────────────
    print("Running inference...")
    resp = post_image(f"{base_url}/infer/{model}", image_path, image_bytes)
    if resp.status_code != 200:
        sys.exit(f"Inference failed ({resp.status_code}): {resp.text}")

    result = resp.json()

    json_path = out_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved  : {json_path}")

    # Pretty-print metrics to terminal
    print()
    print("── Metrics ─────────────────────────────")
    for k, v in result["metrics"].items():
        print(f"  {k:<40s} {v:.4f}°")

    print()
    print("── Keypoints (all 27) ──────────────────")
    for kp in result["keypoints"]:
        print(f"  [{kp['joint_id']:2d}]  x={kp['x']:7.2f}  y={kp['y']:7.2f}  conf={kp['confidence']:.4f}")

    # ── 2. Visualize (axes) ────────────────────────────────────────────────
    print()
    print("Running visualize...")
    resp = post_image(f"{base_url}/infer/{model}/visualize", image_path, image_bytes)
    if resp.status_code != 200:
        sys.exit(f"Visualize failed ({resp.status_code}): {resp.text}")

    vis_path = out_dir / f"{stem}_visualize.png"
    vis_path.write_bytes(resp.content)
    print(f"Saved  : {vis_path}")

    # ── 3. Landmarks (points only) ─────────────────────────────────────────
    print("Running landmarks...")
    resp = post_image(f"{base_url}/infer/{model}/landmarks", image_path, image_bytes)
    if resp.status_code != 200:
        sys.exit(f"Landmarks failed ({resp.status_code}): {resp.text}")

    lm_path = out_dir / f"{stem}_landmarks.png"
    lm_path.write_bytes(resp.content)
    print(f"Saved  : {lm_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
