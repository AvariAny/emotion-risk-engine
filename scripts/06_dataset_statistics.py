from pathlib import Path
import pandas as pd

DATASET = Path("data/processed/emotion_risk_dataset.csv")
REPORTS = Path("reports")

REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(DATASET)

print("=" * 60)
print("ESTADÍSTICAS DEL DATASET")
print("=" * 60)

# =====================================================
# Información básica
# =====================================================

total = len(df)

unique = df["text"].nunique()

duplicates = total - unique

word_counts = df["text"].astype(str).str.split().apply(len)

char_counts = df["text"].astype(str).str.len()

report = []

report.append("=" * 60)
report.append("EMOTION RISK DATASET")
report.append("=" * 60)
report.append("")
report.append(f"Total ejemplos           : {total}")
report.append(f"Mensajes únicos          : {unique}")
report.append(f"Duplicados               : {duplicates}")
report.append("")
report.append("LONGITUD")
report.append(f"Palabras promedio        : {word_counts.mean():.2f}")
report.append(f"Palabras mínimas         : {word_counts.min()}")
report.append(f"Palabras máximas         : {word_counts.max()}")
report.append("")
report.append(f"Caracteres promedio      : {char_counts.mean():.2f}")
report.append(f"Caracteres mínimos       : {char_counts.min()}")
report.append(f"Caracteres máximos       : {char_counts.max()}")
report.append("")
report.append("=" * 60)
report.append("DISTRIBUCIÓN POR LABEL")
report.append("=" * 60)

print("\nDistribución por label:\n")

label_counts = df["label"].value_counts().sort_index()

for label, count in label_counts.items():

    percentage = count / total * 100

    line = f"Label {label}: {count} ({percentage:.2f}%)"

    report.append(line)

    print(line)

report.append("")
report.append("=" * 60)
report.append("DISTRIBUCIÓN POR TOPIC")
report.append("=" * 60)

print("\nDistribución por topic:\n")

topic_counts = df.groupby("topic").size().sort_values(ascending=False)

for topic, count in topic_counts.items():

    line = f"{topic:<18} {count}"

    report.append(line)

    print(line)

report_file = REPORTS / "dataset_statistics.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n" + "=" * 60)
print("Reporte generado:")
print(report_file)