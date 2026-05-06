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


def model_kind(name: str) -> str:
    """
    Classify a model file name into an inference pipeline kind.

    'kneeap' → knee AP (single-joint) model with 30 landmarks, JLCA metric.
    'cpak'   → full-leg model with 27 landmarks, LDFA/MPTA metrics.
    """
    return "kneeap" if name.lower().startswith("kneeap") else "cpak"


def list_models() -> list[dict]:
    return [
        {"name": name, "kind": model_kind(name)}
        for name in sorted(AVAILABLE_MODELS.keys())
    ]


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
            f"Available: {sorted(AVAILABLE_MODELS.keys())}"
        )

    if model_name not in _sessions:
        _sessions[model_name] = ort.InferenceSession(
            str(AVAILABLE_MODELS[model_name]),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    return _sessions[model_name]
