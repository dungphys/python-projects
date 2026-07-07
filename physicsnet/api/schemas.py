from pydantic import BaseModel, Field
from typing import Literal

class ParticleFeatures(BaseModel):
    energy:     float = Field(..., gt=0,   example=85.3)
    momentum_x: float = Field(...,         example=12.1)
    momentum_y: float = Field(...,         example=-5.4)
    momentum_z: float = Field(...,         example=30.2)
    charge:     float = Field(..., ge=-1, le=1, example=-1.0)
    hit_0:      int   = Field(..., ge=0,   example=8)
    hit_1:      int   = Field(..., ge=0,   example=9)
    hit_2:      int   = Field(..., ge=0,   example=7)
    hit_3:      int   = Field(..., ge=0,   example=10)

class PredictionResponse(BaseModel):
    particle:      str
    confidence:    float
    probabilities: dict[str, float]
    model_version: str