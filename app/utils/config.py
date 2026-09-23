"""
Configuración global del Emotion Risk Engine.
"""

import os
from pathlib import Path

# ============================================================
# Información de la API
# ============================================================

APP_NAME = "Emotion Risk Engine API"
APP_DESCRIPTION = (
    "REST API para la detección de riesgo emocional mediante BETO."
)
VERSION = "2.0.0"

# ============================================================
# Directorios del proyecto y Modelo
# ============================================================

# app/utils/config.py -> raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

_LOCAL_MODEL_DIR = BASE_DIR / "models" / "emotion-risk-beto"

# Si la carpeta local existe la usa; si no (en producción/Render), apunta a Hugging Face Hub
MODEL_DIR = os.getenv(
    "MODEL_DIR",
    str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else "avarixo/emotion-risk-beto",
)

# ============================================================
# Modelo
# ============================================================

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

MAX_LENGTH = 128

NUM_LABELS = 4

# ============================================================
# Etiquetas
# ============================================================

LABELS = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo moderado",
    3: "Riesgo alto",
}

# ============================================================
# Dispositivo
# ============================================================

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"