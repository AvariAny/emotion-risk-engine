from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_PATH = "models/emotion_model"

print("Cargando modelo...")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

labels = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo alto",
    3: "Riesgo crítico"
}

while True:
    texto = input("\nEscribe un texto (o salir): ")

    if texto.lower() == "salir":
        break

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    print(f"\nPredicción: {labels[pred]}")