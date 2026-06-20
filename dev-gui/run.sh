#!/usr/bin/env bash
# dev-gui başlatıcı (mac/linux). Projenin .venv'ini kullanır, gerekirse
# pywebview'i kurar ve masaüstü penceresini açar.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Proje sanal ortamı bulunamadı: $PY" >&2
  exit 1
fi

# pywebview yoksa bağımlılıkları kur (bir kerelik).
if ! "$PY" -c "import webview" >/dev/null 2>&1; then
  echo "Bağımlılıklar kuruluyor (pywebview)..."
  "$PY" -m pip install -r "$ROOT/dev-gui/requirements.txt"
fi

exec "$PY" "$ROOT/dev-gui/app.py" "$@"
