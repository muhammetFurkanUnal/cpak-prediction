#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
