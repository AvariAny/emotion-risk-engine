from pathlib import Path

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


MODEL_PATH = Path("models/emotion_model")

LABELS = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo alto",
    3: "Riesgo crítico",
}


class EmotionRiskModel:
    def __init__(self):
        print("🧠 Cargando modelo...")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            str(MODEL_PATH)
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            str(MODEL_PATH)
        )

        self.model.eval()

        print("✅ Modelo listo.")

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        label = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0][label].item()

        return {
            "label": label,
            "risk": LABELS[label],
            "confidence": round(confidence, 4),
        }