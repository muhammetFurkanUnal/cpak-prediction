"""
Pydantic request / response schemas for the cpak inference API.
"""

from typing import Literal, Union

from pydantic import BaseModel


class Keypoint(BaseModel):
    joint_id: int
    x: float
    y: float
    confidence: float


# ── cpak (full leg) ───────────────────────────────────────────────────────────

class CpakAngles(BaseModel):
    femur_mech_angle_notch: float   # LDFA
    tibia_mech_angle_inter: float   # MPTA


class CpakResponse(BaseModel):
    kind: Literal["cpak"] = "cpak"
    model: str
    keypoints: list[Keypoint]
    metrics: CpakAngles


# ── kneeap (knee AP) ──────────────────────────────────────────────────────────

class KneeApAngles(BaseModel):
    jlca: float


class KneeApResponse(BaseModel):
    kind: Literal["kneeap"] = "kneeap"
    model: str
    keypoints: list[Keypoint]
    metrics: KneeApAngles


InferenceResponse = Union[CpakResponse, KneeApResponse]


# ── dual-knee responses ───────────────────────────────────────────────────────
# All keypoint coordinates are returned in the ORIGINAL image's pixel space
# (the right-side keypoints have already been offset by split_x server-side),
# so the frontend can draw both overlays on the un-cropped image directly.

class CpakKneeSide(BaseModel):
    keypoints: list[Keypoint]
    metrics: CpakAngles


class KneeApKneeSide(BaseModel):
    keypoints: list[Keypoint]
    metrics: KneeApAngles


class DualDetection(BaseModel):
    """Bounding boxes from the YOLO knee detector, in original image coords."""
    left:  list[float]   # [x1, y1, x2, y2]
    right: list[float]


class DualCpakResponse(BaseModel):
    kind: Literal["cpak"] = "cpak"
    mode: Literal["dual"] = "dual"
    model: str
    split_x: int
    detection: DualDetection
    left:  CpakKneeSide
    right: CpakKneeSide


class DualKneeApResponse(BaseModel):
    kind: Literal["kneeap"] = "kneeap"
    mode: Literal["dual"] = "dual"
    model: str
    split_x: int
    detection: DualDetection
    left:  KneeApKneeSide
    right: KneeApKneeSide


DualInferenceResponse = Union[DualCpakResponse, DualKneeApResponse]
