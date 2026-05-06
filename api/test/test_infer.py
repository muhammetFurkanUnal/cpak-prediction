#!/usr/bin/env python3
"""
Test: POST /infer/{model_name}

Usage
-----
python3 api/test/test_infer.py [base_url] [image_path] [model_name]

Defaults
--------
base_url   : http://localhost:8000
image_path : data/samples-img/split-postop/4024.r.jpg  (relative to project root)
model_name : first model returned by /models
"""

import json
import sys
from pathlib import Path
import requests

BASE_URL   = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE_PATH = sys.argv[2] if len(sys.argv) > 2 else str(
    Path(__file__).parent.parent.parent / "data/samples-img/split-postop/4024.r.jpg"
)

# Pick model. /models returns [{name, kind}] (older builds may return [str]).
if len(sys.argv) > 3:
    MODEL = sys.argv[3]
else:
    models_resp = requests.get(f"{BASE_URL}/models")
    models_resp.raise_for_status()
    first = models_resp.json()["models"][0]
    MODEL = first["name"] if isinstance(first, dict) else first

print(f"Image  : {IMAGE_PATH}")
print(f"Model  : {MODEL}")
print(f"URL    : {BASE_URL}/infer/{MODEL}")
print()

with open(IMAGE_PATH, "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/infer/{MODEL}",
        files={"image": (Path(IMAGE_PATH).name, f, "image/jpeg")},
    )

print(f"Status : {resp.status_code}")

assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

body = resp.json()
assert body["model"] == MODEL
assert "metrics" in body
kind = body.get("kind", "cpak")
expected_kp = 30 if kind == "kneeap" else 27
assert len(body["keypoints"]) == expected_kp, \
    f"Expected {expected_kp} keypoints for kind={kind}, got {len(body['keypoints'])}"
if kind == "kneeap":
    assert "jlca" in body["metrics"]
else:
    assert "femur_mech_angle_notch" in body["metrics"]
    assert "tibia_mech_angle_inter" in body["metrics"]

print("Keypoints (first 5):")
for kp in body["keypoints"][:5]:
    print(f"  [{kp['joint_id']:2d}] x={kp['x']:.1f}  y={kp['y']:.1f}  conf={kp['confidence']:.4f}")

print()
print("Metrics:")
for k, v in body["metrics"].items():
    print(f"  {k}: {v:.4f}")

print()
print("PASS")
