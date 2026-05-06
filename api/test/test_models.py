#!/usr/bin/env python3
"""Test: GET /models"""

import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

resp = requests.get(f"{BASE_URL}/models")
print(f"Status  : {resp.status_code}")
print(f"Body    : {resp.json()}")

assert resp.status_code == 200, "Expected 200"
models = resp.json().get("models", [])
assert isinstance(models, list), "Expected 'models' to be a list"
assert len(models) > 0, "No models returned — check notebooks/out/models/"
for m in models:
    assert isinstance(m, dict) and "name" in m and "kind" in m, \
        f"Each model entry must be {{name, kind}}, got {m!r}"
    assert m["kind"] in ("cpak", "kneeap"), f"Unknown kind: {m['kind']!r}"
print("PASS")
