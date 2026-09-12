# evaluate_model.py
# Evaluación del modelo Emotion Risk Engine

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
)

# ==========================
# Configuración
# ==========================

DATASET_PATH = Path("data/processed/emotion_risk_dataset.csv")
MODEL_PATH = Path("models/emotion_model")

LABELS = {
    0: "Sin riesgo",
    1: "Riesgo bajo",
    2: "Riesgo alto",
    3: "Riesgo crítico",
}

# ==========================
# Cargar dataset
# ==========================

print("=" * 60)
print("Emotion Risk Engine - Evaluación")
print("=" * 60)

print("\n📂 Cargando dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"\nTotal de ejemplos: {len(df)}")

print("\nDistribución de etiquetas:")

print(df["label"].value_counts().sort_index())

# ==========================
# Train / Validation
# ==========================

print("\n✂️ Separando conjunto de validación...")

_, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

print(f"Ejemplos de validación: {len(val_df)}")

# Guardamos una copia con texto para analizar errores
val_texts = val_df["text"].reset_index(drop=True)

# ==========================
# Dataset Hugging Face
# ==========================

val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# ==========================
# Tokenizer
# ==========================

print("\n🔤 Cargando tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


print("\n⚙️ Tokenizando...")

val_dataset = val_dataset.map(tokenize, batched=True)

# ==========================
# Preparar columnas
# ==========================

val_dataset = val_dataset.remove_columns(["text"])
val_dataset = val_dataset.rename_column("label", "labels")
val_dataset.set_format("torch")

# ==========================
# Modelo
# ==========================

print("\n🧠 Cargando modelo entrenado...")

model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))

trainer = Trainer(model=model)

# ==========================
# Predicciones
# ==========================

print("\n🚀 Realizando predicciones...")

predictions = trainer.predict(val_dataset)

pred_logits = predictions.predictions

probabilities = np.exp(pred_logits) / np.sum(
    np.exp(pred_logits),
    axis=1,
    keepdims=True,
)

pred_labels = np.argmax(pred_logits, axis=1)

confidences = probabilities.max(axis=1)

true_labels = np.array(val_dataset["labels"])

# ==========================
# Métricas generales
# ==========================

accuracy = accuracy_score(true_labels, pred_labels)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    pred_labels,
    average="weighted",
)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ==========================
# Reporte de clasificación
# ==========================

print("\n" + "=" * 60)
print("REPORTE POR CLASE")
print("=" * 60)

print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=[
            LABELS[0],
            LABELS[1],
            LABELS[2],
            LABELS[3],
        ],
        digits=4,
    )
)

# ==========================
# Matriz de confusión bonita
# ==========================

print("\n" + "=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

cm = confusion_matrix(true_labels, pred_labels)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Real: Sin riesgo",
        "Real: Riesgo bajo",
        "Real: Riesgo alto",
        "Real: Riesgo crítico",
    ],
    columns=[
        "Pred: Sin riesgo",
        "Pred: Riesgo bajo",
        "Pred: Riesgo alto",
        "Pred: Riesgo crítico",
    ],
)

print(cm_df)

# ==========================
# Distribución de predicciones
# ==========================

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE PREDICCIONES")
print("=" * 60)

pred_distribution = pd.Series(pred_labels).value_counts().sort_index()

for label, count in pred_distribution.items():
    print(f"{LABELS[label]:18} : {count}")

# ==========================
# Ejemplos mal clasificados
# ==========================

print("\n" + "=" * 60)
print("EJEMPLOS MAL CLASIFICADOS")
print("=" * 60)

errors = 0

for i in range(len(true_labels)):

    if true_labels[i] != pred_labels[i]:

        print(f"\nEjemplo #{errors+1}")

        print("-" * 50)

        print(f"Texto:\n{val_texts.iloc[i]}")

        print(f"\nEtiqueta real : {LABELS[true_labels[i]]}")

        print(f"Predicción    : {LABELS[pred_labels[i]]}")

        print(f"Confianza     : {confidences[i]*100:.2f}%")

        errors += 1

        if errors == 20:
            break

# ==========================
# Resumen
# ==========================

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Total ejemplos           : {len(df)}")
print(f"Validación               : {len(val_df)}")
print(f"Errores mostrados        : {errors}")
print(f"Accuracy                 : {accuracy:.4f}")
print(f"F1-score                 : {f1:.4f}")

print("\n✅ Evaluación terminada.")