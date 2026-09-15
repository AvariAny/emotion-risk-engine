import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ============================================================
# Configuración
# ============================================================

MODEL_DIR = "models/emotion-risk-beto"

LABELS = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo moderado",
    3: "Riesgo alto"
}

# ============================================================
# Cargar modelo
# ============================================================

print("=" * 60)
print("Cargando modelo...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

# ============================================================
# Predicción
# ============================================================

while True:

    print("\n")

    text = input("Escribe un mensaje ('salir' para terminar): ")

    if text.lower() == "salir":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    print("\nPredicción")
    print("-----------------------------")
    print(f"Label: {prediction}")
    print(f"Clase: {LABELS[prediction]}")