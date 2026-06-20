#!/usr/bin/env bash
# Bootstrap a local venv and launch tagimg. Works on macOS and Linux.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet Pillow
fi

exec "$VENV/bin/python" "$DIR/tagimg.py" "$@"
