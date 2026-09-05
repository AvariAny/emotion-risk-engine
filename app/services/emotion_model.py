from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../models/emotion_model"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)[0]

    probabilities = probabilities.tolist()

    print("\nEmotion probabilities:\n")

    for i, value in enumerate(probabilities):
        print(f"{i:2d}: {value:.4f}")

    return probabilities