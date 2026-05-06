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
