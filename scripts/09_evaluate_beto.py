import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Configuración
# ============================================================

MODEL_DIR = "models/emotion-risk-beto"
TEST_FILE = "data/processed/test.csv"

REPORT_DIR = "reports"

MAX_LENGTH = 128

os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Modelo
# ============================================================

print("=" * 60)
print("Cargando modelo...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

# ============================================================
# Dataset
# ============================================================

test_df = pd.read_csv(TEST_FILE)

print(f"Test examples: {len(test_df)}")

dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)

# ============================================================
# Tokenización
# ============================================================

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ============================================================
# Predicción
# ============================================================

print("\nRealizando predicciones...\n")

predictions = trainer.predict(dataset)

preds = np.argmax(
    predictions.predictions,
    axis=1
)

labels = predictions.label_ids

# ============================================================
# Métricas
# ============================================================

accuracy = accuracy_score(labels, preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="macro"
)

print("=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# ============================================================
# Classification Report
# ============================================================

report = classification_report(
    labels,
    preds,
    digits=4
)

print("\n")
print(report)

with open(
    os.path.join(REPORT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ============================================================
# Matriz de confusión
# ============================================================

cm = confusion_matrix(labels, preds)

disp = ConfusionMatrixDisplay(cm)

fig, ax = plt.subplots(figsize=(7,7))

disp.plot(ax=ax)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("\nReporte guardado.")

print("Matriz guardada.")