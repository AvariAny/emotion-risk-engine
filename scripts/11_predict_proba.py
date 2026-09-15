import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Configuración
# ============================================================

MODEL_PATH = "models/emotion-risk-beto"

LABELS = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo moderado",
    3: "Riesgo alto"
}

# ============================================================
# Cargar modelo
# ============================================================

print("Cargando modelo...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# ============================================================
# Predicción con probabilidades
# ============================================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

    prediction = torch.argmax(probs).item()

    print("\nMensaje:")
    print(text)

    print("\nProbabilidades")

    for i, p in enumerate(probs):

        print(f"Label {i} ({LABELS[i]}): {p.item()*100:.2f}%")

    print("\nPredicción final")
    print("----------------------------")
    print("Label:", prediction)
    print("Clase:", LABELS[prediction])

# ============================================================
# Consola
# ============================================================

while True:

    text = input("\nEscribe un mensaje ('salir' para terminar): ")

    if text.lower() == "salir":
        break

    predict(text)