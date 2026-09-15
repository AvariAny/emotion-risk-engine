import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ============================================================
# Configuración
# ============================================================

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

TRAIN_FILE = "data/processed/train.csv"
VALID_FILE = "data/processed/validation.csv"

OUTPUT_DIR = "models/emotion-risk-beto"

NUM_LABELS = 4

MAX_LENGTH = 128

SEED = 42

# ============================================================
# Reproducibilidad
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# Cargar tokenizer
# ============================================================

print("=" * 60)
print("Cargando tokenizer BETO...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# Cargar CSVs
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(valid_df)}")

# ============================================================
# Convertir a Dataset de Hugging Face
# ============================================================

train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
valid_dataset = Dataset.from_pandas(valid_df, preserve_index=False)

# ============================================================
# Tokenización
# ============================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

# ============================================================
# Renombrar etiqueta
# ============================================================

train_dataset = train_dataset.rename_column("label", "labels")
valid_dataset = valid_dataset.rename_column("label", "labels")

# ============================================================
# Modelo
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# ============================================================
# Métricas
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================
# Argumentos de entrenamiento
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",          # <-- cambiado de evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,     # <-- cambiado de tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# Entrenamiento
# ============================================================

trainer.train()

# ============================================================
# Evaluación final
# ============================================================

metrics = trainer.evaluate()
print("Métricas finales:")
print(metrics)

# ============================================================
# Guardar modelo y tokenizer
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Modelo guardado en: {OUTPUT_DIR}")