from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/merged_topics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_topics = 0
total_examples = 0

for label_dir in sorted(RAW_DIR.glob("label*")):

    output_label = OUTPUT_DIR / label_dir.name
    output_label.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {label_dir.name.upper()} =====")

    for topic_dir in sorted(label_dir.iterdir()):

        if not topic_dir.is_dir():
            continue

        csv_files = list(topic_dir.glob("*.csv"))

        if not csv_files:
            continue

        dfs = []

        for csv_file in csv_files:

            df = pd.read_csv(csv_file)

            if "text" not in df.columns or "label" not in df.columns:
                print(f"⚠ {csv_file} ignorado.")
                continue

            df["label"] = df["label"].astype(int)

            dfs.append(df)

        if not dfs:
            continue

        merged = pd.concat(dfs, ignore_index=True)

        output_file = output_label / f"{topic_dir.name}.csv"

        merged.to_csv(output_file, index=False)

        print(
            f"{topic_dir.name:<18}"
            f"{len(csv_files)} archivos"
            f" -> {len(merged)} ejemplos"
        )

        total_topics += 1
        total_examples += len(merged)

print("\n===================================")
print("Merge completado")
print("===================================")
print(f"Topics procesados : {total_topics}")
print(f"Ejemplos totales  : {total_examples}")