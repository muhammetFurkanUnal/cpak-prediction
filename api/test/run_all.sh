#!/usr/bin/env bash
# Run all endpoint tests sequentially.
#
# Usage
# -----
# ./api/test/run_all.sh [base_url] [image_path]
#
# Defaults
# --------
# base_url   : http://localhost:8000
# image_path : data/samples-img/split-postop/4024.r.jpg

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

BASE_URL="${1:-http://localhost:8000}"
IMAGE_PATH="${2:-$PROJECT_ROOT/data/samples-img/split-postop/4024.r.jpg}"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local script="$2"
    shift 2
    echo "────────────────────────────────────"
    echo "TEST: $name"
    if python3 "$script" "$@"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $name"
    fi
    echo
}

run_test "health"     api/test/test_health.py    "$BASE_URL"
run_test "models"     api/test/test_models.py    "$BASE_URL"
run_test "infer"      api/test/test_infer.py     "$BASE_URL" "$IMAGE_PATH"
run_test "visualize"  api/test/test_visualize.py "$BASE_URL" "$IMAGE_PATH"
run_test "landmarks"  api/test/test_landmarks.py "$BASE_URL" "$IMAGE_PATH"

echo "════════════════════════════════════"
echo "Results: $PASS passed, $FAIL failed"

[ "$FAIL" -eq 0 ]
