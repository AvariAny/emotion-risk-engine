"""
Schemas (Pydantic) del Emotion Risk Engine API.
"""

from datetime import datetime
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
# History
# ============================================================

class PredictionHistory(BaseModel):
    """
    Representación de una predicción almacenada en la base de datos.
    """

    id: int
    text: str
    risk: int
    confidence: float
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


# ============================================================
# Update
# ============================================================

class PredictionUpdate(BaseModel):
    """
    Campos editables de una predicción existente.
    """

    risk: int


# ============================================================
# Stats
# ============================================================

class PredictionStats(BaseModel):
    """
    Estadísticas agregadas del historial de predicciones.
    """

    total_predictions: int

    average_confidence: float

    risk_distribution: dict[int, int]


# ============================================================
# Alert
# ============================================================

class AlertResponse(BaseModel):
    """
    Representación de una predicción de alto riesgo.
    """

    id: int
    text: str
    risk: int
    confidence: float
    created_at: datetime

    model_config = {
        "from_attributes": True
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