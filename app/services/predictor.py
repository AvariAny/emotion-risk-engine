import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy.orm import Session

from app.utils.config import LABELS, MODEL_DIR, MAX_LENGTH, DEVICE
from app.utils.text_normalizer import normalize_text
from app.database.models import Prediction


class EmotionPredictor:
    def __init__(self):
        print(f"Cargando modelo BETO desde {MODEL_DIR} en memoria...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"Modelo cargado exitosamente ({DEVICE})")

    def predict(self, text: str, db: Session = None):
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

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        label = int(torch.argmax(probs))
        confidence = round(float(probs[label]), 4)
        probabilities = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}

        if db is not None:
            try:
                row = Prediction(text=text, risk=label, confidence=confidence)
                db.add(row)
                db.commit()
                db.refresh(row)
            except Exception as e:
                print(f"Advertencia: no se pudo registrar en BD: {e}")
                db.rollback()

        return {
            "label": label,
            "class": LABELS.get(label, "Desconocido"),
            "confidence": confidence,
            "probabilities": probabilities,
        }