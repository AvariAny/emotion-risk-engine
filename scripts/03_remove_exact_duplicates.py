from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/merged_topics")
OUTPUT_DIR = Path("data/cleaned_topics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ELIMINANDO DUPLICADOS EXACTOS")
print("=" * 60)

total_original = 0
total_final = 0
total_duplicates = 0

for label_dir in sorted(INPUT_DIR.glob("label*")):

    output_label = OUTPUT_DIR / label_dir.name
    output_label.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {label_dir.name.upper()} =====")

    for csv_file in sorted(label_dir.glob("*.csv")):

        df = pd.read_csv(csv_file)

        original = len(df)

        # Eliminar duplicados exactos por el texto
        df_clean = df.drop_duplicates(subset=["text"], keep="first")

        final = len(df_clean)

        duplicates = original - final

        total_original += original
        total_final += final
        total_duplicates += duplicates

        output_file = output_label / csv_file.name
        df_clean.to_csv(output_file, index=False)

        print(
            f"{csv_file.stem:<18}"
            f"Original: {original:<4}"
            f" Duplicados: {duplicates:<3}"
            f" Final: {final}"
        )

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(f"Ejemplos originales : {total_original}")
print(f"Duplicados removidos: {total_duplicates}")
print(f"Ejemplos finales    : {total_final}")