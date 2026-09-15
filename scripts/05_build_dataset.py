from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/semantic_cleaned")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "emotion_risk_dataset.csv"

print("=" * 60)
print("CONSTRUYENDO DATASET FINAL")
print("=" * 60)

dfs = []

topic_stats = []

for label_dir in sorted(INPUT_DIR.glob("label*")):

    label_name = label_dir.name

    for csv_file in sorted(label_dir.glob("*.csv")):

        df = pd.read_csv(csv_file)

        # Agregar metadata útil
        df["topic"] = csv_file.stem
        df["label_name"] = label_name

        dfs.append(df)

        topic_stats.append({
            "label": label_name,
            "topic": csv_file.stem,
            "examples": len(df)
        })

dataset = pd.concat(dfs, ignore_index=True)

# Mezclar aleatoriamente
dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

dataset.to_csv(OUTPUT_FILE, index=False)

# ==========================
# Estadísticas
# ==========================

print("\nDataset construido correctamente.\n")

print(f"Total ejemplos : {len(dataset)}")

print("\nDistribución por label\n")

print(
    dataset["label"]
    .value_counts()
    .sort_index()
)

print("\nEjemplos por topic\n")

topic_df = pd.DataFrame(topic_stats)

print(
    topic_df
    .groupby("label")["examples"]
    .sum()
)

print(f"\nArchivo generado:\n{OUTPUT_FILE}")