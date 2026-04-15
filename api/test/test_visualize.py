#!/usr/bin/env python3
"""
Test: POST /infer/{model_name}/visualize

Saves the returned JPEG to /tmp/cpak_visualize.jpg so you can inspect it.

Usage
-----
python3 api/test/test_visualize.py [base_url] [image_path] [model_name]
"""

import sys
from pathlib import Path
import requests

BASE_URL   = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE_PATH = sys.argv[2] if len(sys.argv) > 2 else str(
    Path(__file__).parent.parent.parent / "data/samples-img/split-postop/4024.r.jpg"
)
OUT_PATH   = "/tmp/cpak_visualize.jpg"

if len(sys.argv) > 3:
    MODEL = sys.argv[3]
else:
    MODEL = requests.get(f"{BASE_URL}/models").json()["models"][0]

print(f"Image  : {IMAGE_PATH}")
print(f"Model  : {MODEL}")
print(f"URL    : {BASE_URL}/infer/{MODEL}/visualize")
print()

with open(IMAGE_PATH, "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/infer/{MODEL}/visualize",
        files={"image": (Path(IMAGE_PATH).name, f, "image/jpeg")},
    )

print(f"Status       : {resp.status_code}")
print(f"Content-Type : {resp.headers.get('content-type')}")

assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
assert "image/jpeg" in resp.headers.get("content-type", ""), "Expected JPEG response"
assert len(resp.content) > 0, "Empty image body"

with open(OUT_PATH, "wb") as f:
    f.write(resp.content)

print(f"Saved        : {OUT_PATH}  ({len(resp.content):,} bytes)")
print()
print("PASS")
