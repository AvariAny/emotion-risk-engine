"""
Predictor del Emotion Risk Engine mediante llamada directa a Hugging Face Hub.
"""

import os
import json
import time
import requests
from sqlalchemy.orm import Session

from app.utils.config import LABELS
from app.utils.text_normalizer import normalize_text
from app.database.models import Prediction

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME", "avarixo/emotion-risk-beto")
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"


class EmotionPredictor:

    def __init__(self):
        print(f"Configurando cliente HTTP hacia: {API_URL}")
        # Añadimos Content-Type explícito y Accept para forzar respuesta JSON clara
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if HF_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_TOKEN}"
        else:
            print("ADVERTENCIA: HF_TOKEN no está definido en el entorno.")

    def _query_hf_api(self, text: str, max_retries: int = 4):
        """Envía el texto a Hugging Face y reintenta si el modelo está cargando."""
        payload = {
            "inputs": text,
            "parameters": {},
            "options": {"wait_for_model": True, "use_cache": True},
        }

        last_response = None

        for attempt in range(1, max_retries + 1):
            print(f"[HF] Intento {attempt}/{max_retries} -> POST {API_URL}")
            print(f"[HF] Payload enviado: {json.dumps(payload, ensure_ascii=False)[:300]}")

            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=45,
            )

            print(f"[HF] Status: {response.status_code}")
            print(f"[HF] Content-Type: {response.headers.get('content-type')}")

            if response.status_code == 200:
                return response.json()

            # ---- Imprimir SIEMPRE el cuerpo crudo del error para depuración ----
            print(f"[HF] Cuerpo de respuesta: {response.text[:1000]}")

            # Intentar parsear JSON de forma segura
            data = {}
            if "application/json" in (response.headers.get("content-type") or ""):
                try:
                    data = response.json()
                except Exception as e:
                    print(f"[HF] No se pudo parsear JSON: {e}")

            # Si el modelo se está cargando, esperamos y reintentamos
            if response.status_code == 503 and "estimated_time" in data:
                wait_time = min(float(data.get("estimated_time", 10)), 20)
                print(f"[HF] Modelo inicializándose. Esperando {wait_time:.1f}s...")
                time.sleep(wait_time)
                last_response = response
                continue

            # 401 / 403: token inválido o sin permisos
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"[HF] Autenticación fallida ({response.status_code}). "
                    f"Revisa HF_TOKEN. Detalle: {response.text}"
                )

            # 404: modelo inexistente o mal escrito
            if response.status_code == 404:
                raise RuntimeError(
                    f"[HF] Modelo no encontrado en '{API_URL}'. "
                    f"Verifica MODEL_NAME. Detalle: {response.text}"
                )

            # 400 / 422: payload mal formado -> MOSTRAR DETALLE
            if response.status_code in (400, 422):
                raise RuntimeError(
                    f"[HF] Payload rechazado ({response.status_code}). "
                    f"Detalle devuelto por HF: {response.text}"
                )

            # Otros códigos: reintentamos una vez más si es 5xx
            if 500 <= response.status_code < 600 and attempt < max_retries:
                last_response = response
                time.sleep(2)
                continue

            response.raise_for_status()

        raise RuntimeError(
            f"Hugging Face API falló tras {max_retries} intentos. "
            f"Último status: {getattr(last_response, 'status_code', 'N/A')}, "
            f"detalle: {getattr(last_response, 'text', 'sin cuerpo')}"
        )

    def predict(self, text: str, db: Session = None):
        # 1) Normalización
        original_text = text
        text = normalize_text(text)

        if not text or not text.strip():
            text = (original_text or "").strip()

        if not text:
            return {
                "label": -1,
                "class": None,
                "confidence": 0.0,
                "probabilities": {LABELS[i]: 0.0 for i in range(len(LABELS))},
            }

        # 2) Petición a HF
        hf_results = self._query_hf_api(text)

        # Normalizar la forma de la respuesta
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

            if "LABEL_" in raw_label:
                try:
                    idx = int(raw_label.replace("LABEL_", ""))
                except ValueError:
                    idx = 0
            elif raw_label.isdigit():
                idx = int(raw_label)
            else:
                idx = next(
                    (k for k, v in LABELS.items() if v.lower() == raw_label.lower()),
                    0,
                )

            if idx in LABELS:
                probabilities[LABELS[idx]] = score

            if score > best_score:
                best_score = score
                best_label = idx

        confidence = best_score

        # 3) Guardado en BD
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