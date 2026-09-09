# train_model.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DATASET_PATH = Path("data/processed/emotion_risk_dataset.csv")
MODEL_PATH = Path("models/emotion_model")

print("Cargando dataset...")
df = pd.read_csv(DATASET_PATH)

print("\nPrimeras filas del dataset:")
print(df.head())

print("\nDistribucion de etiquetas:")
print(df["label"].value_counts())
print(f"\nTotal de ejemplos: {len(df)}")

print("\nDividiendo en entrenamiento y validacion...")
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print(f"Entrenamiento: {len(train_df)} ejemplos")
print(f"Validacion: {len(val_df)} ejemplos")

print("\nConvirtiendo a formato Hugging Face...")
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print("\nCargando tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

print("\nTokenizando datasets...")
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

print("\nPreparando columnas para el modelo...")
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
val_dataset.set_format("torch")

print("\nDataset listo para entrenar!")
print(f"Train features: {train_dataset.column_names}")
print(f"Val features: {val_dataset.column_names}")

print("\nCargando DistilBERT...")
model = DistilBertForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    num_labels=4,
    ignore_mismatched_sizes=True,
)

model.config.problem_type = "single_label_classification"

print("Modelo cargado.")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

print("\nConfigurando entrenamiento...")
training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

print("\nCreando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nIniciando entrenamiento...")
trainer.train()

print("\nGuardando modelo...")
trainer.save_model(str(MODEL_PATH))
tokenizer.save_pretrained(str(MODEL_PATH))

print("Modelo guardado correctamente.")

print("\nResumen del entrenamiento:")
print(f"Modelo guardado en: {MODEL_PATH}")
print(f"Total de ejemplos: {len(df)}")
print(f"Entrenamiento: {len(train_df)}")
print(f"Validacion: {len(val_df)}")
print("Etiquetas: 0 (Sin riesgo), 1 (Riesgo bajo), 2 (Riesgo alto), 3 (Riesgo critico)")