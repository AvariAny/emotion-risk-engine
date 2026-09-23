"""
Predictor del Emotion Risk Engine mediante Hugging Face Inference API.
"""

import os
from sqlalchemy.orm import Session
from huggingface_hub import InferenceClient

from app.utils.config import LABELS
from app.utils.text_normalizer import normalize_text
from app.database.models import Prediction

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME", "avarixo/emotion-risk-beto")


class EmotionPredictor:

    def __init__(self):
        print(f"Conectando con Hugging Face Inference API ({MODEL_ID})...")
        self.client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
        print("Cliente de inferencia configurado exitosamente.")

    def predict(self, text: str, db: Session = None):
        # 1) Normalización de texto
        original_text = text
        text = normalize_text(text)

        if not text or not text.strip():
            text = original_text.strip()

        if not text:
            return {
                "label": -1,
                "class": None,
                "confidence": 0.0,
                "probabilities": {LABELS[i]: 0.0 for i in range(len(LABELS))},
            }

        # 2) Inferencia remota en Hugging Face
        # Retorna una lista de dicts: [{'label': 'LABEL_0', 'score': 0.85}, ...]
        hf_results = self.client.text_classification(text)

        # Mapear resultados a probabilidades
        probabilities = {LABELS[i]: 0.0 for i in range(len(LABELS))}
        best_label = 0
        best_score = 0.0

        for item in hf_results:
            raw_label = item.label  # suele ser "LABEL_0", "LABEL_1" o el nombre de clase
            score = round(float(item.score), 4)

            # Extraer índice numérico si viene en formato LABEL_X
            if "LABEL_" in raw_label:
                idx = int(raw_label.replace("LABEL_", ""))
            elif raw_label.isdigit():
                idx = int(raw_label)
            else:
                # Si el modelo tiene guardados los nombres directos
                idx = next((k for k, v in LABELS.items() if v == raw_label), 0)

            if idx in LABELS:
                probabilities[LABELS[idx]] = score

            if score > best_score:
                best_score = score
                best_label = idx

        confidence = best_score

        # 3) Guardar en base de datos si existe sesión
        if db is not None:
            prediction_row = Prediction(
                text=text,
                risk=best_label,
                confidence=confidence,
            )
            db.add(prediction_row)
            db.commit()
            db.refresh(prediction_row)

        return {
            "label": best_label,
            "class": LABELS.get(best_label, "Desconocido"),
            "confidence": confidence,
            "probabilities": probabilities,
        }