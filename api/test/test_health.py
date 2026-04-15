#!/usr/bin/env python3
"""Test: GET /health"""

import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

resp = requests.get(f"{BASE_URL}/health")
print(f"Status : {resp.status_code}")
print(f"Body   : {resp.json()}")

assert resp.status_code == 200, "Expected 200"
assert resp.json().get("status") == "ok", "Expected {status: ok}"
print("PASS")
