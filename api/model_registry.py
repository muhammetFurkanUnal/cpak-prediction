"""
ONNX model registry with lazy loading and thread-safe caching.

Sessions are loaded on first request and kept in memory.
onnxruntime.InferenceSession is safe to call from multiple threads.
"""

import onnxruntime as ort
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "notebooks" / "out" / "models"

# name → path, populated at import time from whatever .onnx files exist
AVAILABLE_MODELS: dict[str, Path] = {p.stem: p for p in MODELS_DIR.glob("*.onnx")}

_sessions: dict[str, ort.InferenceSession] = {}


def list_models() -> list[str]:
    return sorted(AVAILABLE_MODELS.keys())


def get_session(model_name: str) -> ort.InferenceSession:
    """
    Returns a cached InferenceSession for *model_name*.
    Loads from disk on the first call, returns the cached session afterwards.

    Raises
    ------
    KeyError  if the model name is not in AVAILABLE_MODELS.
    """
    if model_name not in AVAILABLE_MODELS:
        raise KeyError(
            f"Model '{model_name}' not found. "
            f"Available: {list_models()}"
        )

    if model_name not in _sessions:
        _sessions[model_name] = ort.InferenceSession(
            str(AVAILABLE_MODELS[model_name]),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    return _sessions[model_name]
