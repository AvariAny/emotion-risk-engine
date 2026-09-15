"""
Predictor del Emotion Risk Engine.
"""

import torch

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

    def predict(self, text: str):

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

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

        prediction = int(torch.argmax(probabilities))

        return {

            "label": prediction,

            "class": LABELS[prediction],

            "confidence": round(
                float(probabilities[prediction]),
                4
            ),

            "probabilities": {
                LABELS[i]: round(
                    float(probabilities[i]),
                    4
                )
                for i in range(len(LABELS))
            }

        }