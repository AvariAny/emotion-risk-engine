from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================
# Configuración
# ============================

INPUT_DIR = Path("data/cleaned_topics")
OUTPUT_DIR = Path("data/semantic_cleaned")
REPORT_DIR = Path("reports/semantic_duplicates")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

THRESHOLD = 0.95

print("=" * 60)
print("Cargando modelo...")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

total_original = 0
total_removed = 0
total_final = 0

# =======================================================

for label_dir in sorted(INPUT_DIR.glob("label*")):

    output_label = OUTPUT_DIR / label_dir.name
    output_label.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {label_dir.name.upper()} =====")

    for csv_file in sorted(label_dir.glob("*.csv")):

        df = pd.read_csv(csv_file)

        texts = df["text"].astype(str).tolist()

        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        keep = []
        removed = []

        for i in range(len(texts)):

            duplicate = False

            for j in keep:

                sim = cosine_similarity(
                    [embeddings[i]],
                    [embeddings[j]]
                )[0][0]

                if sim >= THRESHOLD:

                    duplicate = True

                    removed.append({
                        "removed_text": texts[i],
                        "kept_text": texts[j],
                        "similarity": round(float(sim),4)
                    })

                    break

            if not duplicate:
                keep.append(i)

        cleaned = df.iloc[keep].reset_index(drop=True)

        report = pd.DataFrame(removed)

        cleaned.to_csv(
            output_label / csv_file.name,
            index=False
        )

        report.to_csv(
            REPORT_DIR / f"{label_dir.name}_{csv_file.stem}.csv",
            index=False
        )

        original = len(df)
        final = len(cleaned)
        removed_count = original - final

        total_original += original
        total_removed += removed_count
        total_final += final

        print(
            f"{csv_file.stem:<18}"
            f"Original:{original:<4}"
            f" Eliminados:{removed_count:<3}"
            f" Final:{final}"
        )

# =======================================================

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Originales : {total_original}")
print(f"Eliminados : {total_removed}")
print(f"Finales    : {total_final}")