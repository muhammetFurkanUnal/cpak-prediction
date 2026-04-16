#!/usr/bin/env python3
"""
Test: POST /infer/{model_name}/visualize

Saves the returned JPEG to api/test/out/cpak_visualize.jpg so you can inspect it.

Usage
-----
python3 api/test/test_visualize.py <image_path> [model_name] [base_url]
"""

import sys
from pathlib import Path
import requests

if len(sys.argv) < 2:
    print("Usage: python3 api/test/test_visualize.py <image_path> [model_name] [base_url]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
BASE_URL   = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
OUT_DIR    = Path(__file__).parent / "out"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH   = str(OUT_DIR / "cpak_visualize.png")

if len(sys.argv) > 2:
    MODEL = sys.argv[2]
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
assert "image/png" in resp.headers.get("content-type", ""), "Expected PNG response"
assert len(resp.content) > 0, "Empty image body"

with open(OUT_PATH, "wb") as f:
    f.write(resp.content)

print(f"Saved        : {OUT_PATH}  ({len(resp.content):,} bytes)")
print()
print("PASS")
