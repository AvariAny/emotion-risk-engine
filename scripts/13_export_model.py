import json
import shutil
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ============================================================
# Configuración
# ============================================================

SOURCE_MODEL = "models/emotion-risk-beto"
PRODUCTION_MODEL = "models/production"

LABELS = {
    "0": "Sin riesgo",
    "1": "Riesgo bajo",
    "2": "Riesgo moderado",
    "3": "Riesgo alto"
}

# ============================================================
# Verificar modelo
# ============================================================

print("=" * 60)
print("EXPORTANDO MODELO PARA PRODUCCIÓN")
print("=" * 60)

source = Path(SOURCE_MODEL)

if not source.exists():
    raise FileNotFoundError(
        f"No existe el modelo entrenado:\n{SOURCE_MODEL}"
    )

# ============================================================
# Limpiar carpeta anterior
# ============================================================

production = Path(PRODUCTION_MODEL)

if production.exists():
    shutil.rmtree(production)

production.mkdir(parents=True)

# ============================================================
# Cargar modelo
# ============================================================

print("\nCargando modelo...")

model = AutoModelForSequenceClassification.from_pretrained(
    SOURCE_MODEL
)

tokenizer = AutoTokenizer.from_pretrained(
    SOURCE_MODEL
)

# ============================================================
# Guardar modelo
# ============================================================

print("Guardando modelo...")

model.save_pretrained(PRODUCTION_MODEL)
tokenizer.save_pretrained(PRODUCTION_MODEL)

# ============================================================
# Metadata
# ============================================================

metadata = {
    "model_name": "Emotion Risk Engine",
    "version": "2.0",
    "language": "Spanish",
    "architecture": "BETO",
    "transformer": "dccuchile/bert-base-spanish-wwm-cased",
    "classes": LABELS,
    "max_length": 128,
    "num_labels": 4,
    "accuracy": 0.9757,
    "precision": 0.9766,
    "recall": 0.9760,
    "f1_score": 0.9760
}

with open(
    production / "model_info.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# Final
# ============================================================

print("\nModelo exportado correctamente.\n")

print("Archivos generados:")

for file in sorted(production.iterdir()):
    print("  •", file.name)

print("\nModelo listo para producción.")