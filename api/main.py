"""
cpak FastAPI inference server.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health                          – liveness check
GET  /models                          – list available ONNX models with their kind
POST /infer/{model_name}              – run inference, return JSON
POST /infer/{model_name}/visualize    – run inference, return annotated image (PNG)
POST /infer/{model_name}/landmarks    – return landmark-only annotated image (PNG)
"""

import sys
from pathlib import Path

# Allow importing notebooks/lib from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.model_registry import get_session, list_models, model_kind
from api.schemas import (
    CpakAngles,
    CpakResponse,
    InferenceResponse,
    KneeApAngles,
    KneeApResponse,
    Keypoint,
)
from notebooks.lib import cpak_inference, kneeap_inference

app = FastAPI(
    title="cpak Inference API",
    description="Ortopedik landmark tespiti ve mekanik açı hesaplama.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_image(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image file to a BGR numpy array."""
    contents = await upload.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid JPEG/PNG file.")
    return img


def _resolve_session(model_name: str):
    try:
        return get_session(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _run_inference(model_name: str, img: np.ndarray):
    """
    Dispatch to the right inference module based on the model kind.

    Returns (kind, coords, metrics_raw).
    """
    kind = model_kind(model_name)
    session = _resolve_session(model_name)
    module = kneeap_inference if kind == "kneeap" else cpak_inference

    heatmap, offset = module.get_predictions(session, img)
    coords = module.extract_coordinates(heatmap, offset)
    if kind == "kneeap":
        metrics_raw = kneeap_inference.compute_kneeap_metrics(coords)
    else:
        metrics_raw = cpak_inference.compute_orthopedic_metrics(coords)
    return kind, coords, metrics_raw


def _build_response(model_name: str, kind: str, coords, metrics_raw) -> InferenceResponse:
    keypoints = [
        Keypoint(joint_id=i, x=float(x), y=float(y), confidence=float(conf))
        for i, (x, y, conf) in enumerate(coords)
    ]
    if kind == "kneeap":
        return KneeApResponse(
            model=model_name,
            keypoints=keypoints,
            metrics=KneeApAngles(jlca=float(metrics_raw["jlca"])),
        )
    return CpakResponse(
        model=model_name,
        keypoints=keypoints,
        metrics=CpakAngles(
            femur_mech_angle_notch=metrics_raw["femur_mech_angle_notch"],
            tibia_mech_angle_inter=metrics_raw["tibia_mech_angle_inter"],
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    # no-store on the entry HTML so users always pick up the latest static asset
    # versions (the JS files themselves can still be cached by the browser).
    return FileResponse(
        str(_FRONTEND / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": list_models()}


@app.post("/infer/{model_name}", response_model=InferenceResponse)
async def infer(model_name: str, image: UploadFile = File(...)):
    img = await _read_image(image)
    kind, coords, metrics_raw = _run_inference(model_name, img)
    return _build_response(model_name, kind, coords, metrics_raw)


@app.post("/infer/{model_name}/visualize")
async def infer_visualize(model_name: str, image: UploadFile = File(...)):
    """Run inference and return an annotated PNG with landmarks and axes drawn."""
    img = await _read_image(image)
    kind, coords, metrics_raw = _run_inference(model_name, img)
    module = kneeap_inference if kind == "kneeap" else cpak_inference
    annotated = module.draw_lines(img, metrics_raw, coords)
    _, buffer = cv2.imencode(".png", annotated)
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/infer/{model_name}/landmarks")
async def infer_landmarks(model_name: str, image: UploadFile = File(...)):
    """Run inference and return an annotated PNG showing only the raw landmark points."""
    img = await _read_image(image)
    kind, coords, _ = _run_inference(model_name, img)
    module = kneeap_inference if kind == "kneeap" else cpak_inference
    annotated = module.visualize_predictions(img, coords, threshold=0.0)
    _, buffer = cv2.imencode(".png", annotated)
    return Response(content=buffer.tobytes(), media_type="image/png")
