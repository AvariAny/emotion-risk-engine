"""
Predictor del Emotion Risk Engine mediante llamada directa a Hugging Face Hub.
"""

import os
import time
import requests
from sqlalchemy.orm import Session

from app.utils.config import LABELS
from app.utils.text_normalizer import normalize_text
from app.database.models import Prediction

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME", "avarixo/emotion-risk-beto")
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"


class EmotionPredictor:

    def __init__(self):
        print(f"Configurando cliente HTTP hacia: {API_URL}")
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    def _query_hf_api(self, text: str, max_retries: int = 4):
        """Envía el texto a Hugging Face y reintenta si el modelo está cargando."""
        payload = {"inputs": text, "options": {"wait_for_model": True}}

        for attempt in range(max_retries):
            response = requests.post(API_URL, headers=self.headers, json=payload, timeout=45)

            if response.status_code == 200:
                return response.json()

            # Si el modelo está cargando (código 503)
            data = response.json() if response.headers.get("content-type") == "application/json" else {}
            if response.status_code == 503 and "estimated_time" in data:
                wait_time = min(float(data.get("estimated_time", 10)), 20)
                print(f"Modelo en Hugging Face inicializándose. Esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

        raise RuntimeError(f"Hugging Face API falló tras {max_retries} intentos: {response.text}")

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

        # 2) Petición a la API
        hf_results = self._query_hf_api(text)

        # La respuesta típica de text-classification es [[{'label': '...', 'score': ...}, ...]]
        if isinstance(hf_results, list) and len(hf_results) > 0 and isinstance(hf_results[0], list):
            items = hf_results[0]
        elif isinstance(hf_results, list):
            items = hf_results
        else:
            items = []

        probabilities = {LABELS[i]: 0.0 for i in range(len(LABELS))}
        best_label = 0
        best_score = 0.0

        for item in items:
            raw_label = str(item.get("label", ""))
            score = round(float(item.get("score", 0.0)), 4)

            # Mapear "LABEL_0", "LABEL_1", "0", "1" o el texto directo
            if "LABEL_" in raw_label:
                idx = int(raw_label.replace("LABEL_", ""))
            elif raw_label.isdigit():
                idx = int(raw_label)
            else:
                idx = next((k for k, v in LABELS.items() if v.lower() == raw_label.lower()), 0)

            if idx in LABELS:
                probabilities[LABELS[idx]] = score

            if score > best_score:
                best_score = score
                best_label = idx

        confidence = best_score

        # 3) Guardar en base de datos si existe sesión
        if db is not None:
            try:
                prediction_row = Prediction(
                    text=text,
                    risk=best_label,
                    confidence=confidence,
                )
                db.add(prediction_row)
                db.commit()
                db.refresh(prediction_row)
            except Exception as e:
                print(f"Advertencia: No se pudo registrar en la base de datos: {e}")
                db.rollback()

        return {
            "label": best_label,
            "class": LABELS.get(best_label, "Desconocido"),
            "confidence": confidence,
            "probabilities": probabilities,
        }