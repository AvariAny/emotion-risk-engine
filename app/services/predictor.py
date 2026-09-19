"""
Predictor del Emotion Risk Engine.
"""

import torch

from sqlalchemy.orm import Session

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from app.utils.config import (
    MODEL_DIR,
    MAX_LENGTH,
    DEVICE,
    LABELS,
)

from app.utils.text_normalizer import normalize_text

from app.database.models import Prediction


class EmotionPredictor:

    def __init__(self):

        print("Cargando modelo BETO...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(f"Modelo cargado correctamente ({DEVICE})")

    def predict(self, text: str, db: Session = None):

        # 1) Normalización del texto ANTES de tokenizar
        original_text = text
        text = normalize_text(text)

        # Fallback: si el normalizador deja el texto vacío,
        # usamos el original para no romper la inferencia.
        if not text or not text.strip():
            text = original_text.strip()

        # Si aún así está vacío, devolvemos algo coherente
        if not text:
            return {
                "label": -1,
                "class": None,
                "confidence": 0.0,
                "probabilities": {
                    LABELS[i]: 0.0 for i in range(len(LABELS))
                },
            }

        # 2) Tokenización
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # 3) Inferencia
        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

        prediction = int(torch.argmax(probabilities))

        confidence = round(
            float(probabilities[prediction]),
            4
        )

        # 4) Persistencia en base de datos (opcional)
        if db is not None:

            prediction_row = Prediction(
                text=text,
                risk=prediction,
                confidence=confidence,
            )

            db.add(prediction_row)
            db.commit()
            db.refresh(prediction_row)

        return {

            "label": prediction,

            "class": LABELS[prediction],

            "confidence": confidence,

            "probabilities": {
                LABELS[i]: round(
                    float(probabilities[i]),
                    4
                )
                for i in range(len(LABELS))
            }

        }