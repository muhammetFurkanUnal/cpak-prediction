"""
cpak FastAPI inference server.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health                          – liveness check
GET  /models                          – list available ONNX models
POST /infer/{model_name}              – run inference, return JSON
POST /infer/{model_name}/visualize    – run inference, return annotated image (JPEG)
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

from api.model_registry import get_session, list_models
from api.schemas import InferenceResponse, Keypoint, OrthopedicAngles
from notebooks.lib.inference import (
    compute_orthopedic_metrics,
    draw_lines,
    extract_coordinates,
    get_predictions,
    visualize_predictions,
)

app = FastAPI(
    title="cpak Inference API",
    description="Ortopedik landmark tespiti ve mekanik açı hesaplama.",
    version="0.1.0",
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


def _build_response(model_name: str, coords, metrics_raw) -> InferenceResponse:
    keypoints = [
        Keypoint(joint_id=i, x=float(x), y=float(y), confidence=float(conf))
        for i, (x, y, conf) in enumerate(coords)
    ]
    return InferenceResponse(
        model=model_name,
        keypoints=keypoints,
        metrics=OrthopedicAngles(
            femur_mech_angle_notch=metrics_raw["femur_mech_angle_notch"],
            tibia_mech_angle_inter=metrics_raw["tibia_mech_angle_inter"],
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(_FRONTEND / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": list_models()}


@app.post("/infer/{model_name}", response_model=InferenceResponse)
async def infer(model_name: str, image: UploadFile = File(...)):
    """
    Run inference on a single X-ray image.

    Returns 27 landmark keypoints and the four ortopedik mekanik açılar.
    """
    try:
        session = get_session(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    img = await _read_image(image)
    heatmap, offset = get_predictions(session, img)
    coords = extract_coordinates(heatmap, offset)
    metrics_raw = compute_orthopedic_metrics(coords)

    return _build_response(model_name, coords, metrics_raw)


@app.post("/infer/{model_name}/visualize")
async def infer_visualize(model_name: str, image: UploadFile = File(...)):
    """
    Run inference and return an annotated JPEG with landmarks and mechanical axes drawn.
    """
    try:
        session = get_session(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    img = await _read_image(image)
    heatmap, offset = get_predictions(session, img)
    coords = extract_coordinates(heatmap, offset)
    metrics_raw = compute_orthopedic_metrics(coords)

    annotated = draw_lines(img, metrics_raw, coords)
    _, buffer = cv2.imencode(".png", annotated)
    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/infer/{model_name}/landmarks")
async def infer_landmarks(model_name: str, image: UploadFile = File(...)):
    """
    Run inference and return an annotated JPEG showing only the raw landmark points
    (without mechanical axis lines).
    """
    try:
        session = get_session(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    img = await _read_image(image)
    heatmap, offset = get_predictions(session, img)
    coords = extract_coordinates(heatmap, offset)

    annotated = visualize_predictions(img, coords, threshold=0.0)
    _, buffer = cv2.imencode(".png", annotated)
    return Response(content=buffer.tobytes(), media_type="image/png")
