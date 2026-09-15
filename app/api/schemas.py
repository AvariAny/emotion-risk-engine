"""
Schemas (Pydantic) del Emotion Risk Engine API.
"""

from typing import Dict

from pydantic import BaseModel, Field


# ============================================================
# Request
# ============================================================

class PredictionRequest(BaseModel):
    """
    Mensaje enviado por el cliente.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Texto que será analizado por el modelo."
    )


# ============================================================
# Response
# ============================================================

class PredictionResponse(BaseModel):
    """
    Respuesta devuelta por la API.
    """

    label: int

    class_name: str = Field(
        ...,
        alias="class"
    )

    confidence: float

    probabilities: Dict[str, float]

    model_config = {
        "populate_by_name": True
    }


# ============================================================
# Health Check
# ============================================================

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    device: str


# ============================================================
# Version
# ============================================================

class VersionResponse(BaseModel):

    app: str

    version: str