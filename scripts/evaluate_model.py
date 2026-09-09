# evaluate_model.py
# Evaluacion del modelo Emotion Risk Engine

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

# Rutas
DATASET_PATH = Path("data/processed/emotion_risk_dataset.csv")
MODEL_PATH = Path("models/emotion_model")

# Cargar dataset
print("Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

print("\nTotal de ejemplos:", len(df))
print("\nDistribucion:")
print(df["label"].value_counts())

# Train / Validation
print("\nSeparando conjunto de validacion...")
_, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

print("Ejemplos de validacion:", len(val_df))

# Dataset HuggingFace
val_dataset = Dataset.from_pandas(val_df)

# Tokenizer
print("\nCargando tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

print("\nTokenizando...")
val_dataset = val_dataset.map(tokenize, batched=True)

# Preparar columnas
val_dataset = val_dataset.remove_columns(["text"])
val_dataset = val_dataset.rename_column("label", "labels")
val_dataset.set_format("torch")

# Cargar el modelo entrenado
print("\nCargando modelo entrenado...")
model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
print("Modelo cargado.")

# Crear Trainer
trainer = Trainer(model=model)

# Realizar predicciones
print("\nRealizando predicciones...")
predictions = trainer.predict(val_dataset)
pred_logits = predictions.predictions
pred_labels = np.argmax(pred_logits, axis=-1)

# Obtener etiquetas reales
true_labels = val_dataset["labels"]

# Calcular métricas
accuracy = accuracy_score(true_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    pred_labels,
    average="weighted",
)

print("\nResultados de evaluacion:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

print("\nReporte de clasificacion:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=["Sin riesgo", "Riesgo bajo", "Riesgo alto", "Riesgo critico"]
))

print("\nMatriz de confusion:")
print(confusion_matrix(true_labels, pred_labels))