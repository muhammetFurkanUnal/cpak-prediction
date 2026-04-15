"""
Pydantic request / response schemas for the cpak inference API.
"""

from pydantic import BaseModel
from typing import Optional


class Keypoint(BaseModel):
    joint_id: int
    x: float
    y: float
    confidence: float


class OrthopedicAngles(BaseModel):
    femur_mech_angle_ax_middle: float
    femur_mech_angle_notch: float
    tibia_mech_angle_ax_middle: float
    tibia_mech_angle_inter: float


class InferenceResponse(BaseModel):
    model: str
    keypoints: list[Keypoint]
    metrics: OrthopedicAngles
