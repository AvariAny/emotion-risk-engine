import pandas as pd

DATASET_PATH = "data/processed/emotion_risk_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 50)
print("DATASET VALIDATION")
print("=" * 50)

# Número de ejemplos
print(f"\nTotal examples: {len(df)}")

# Columnas
print(f"Columns: {list(df.columns)}")

# Valores nulos
print("\nMissing values:")
print(df.isnull().sum())

# Duplicados
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Distribución de etiquetas
print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

# Verificar etiquetas válidas
valid_labels = {0, 1, 2, 3}
invalid = df[~df["label"].isin(valid_labels)]

if len(invalid) == 0:
    print("\nAll labels are valid.")
else:
    print("\nInvalid labels found:")
    print(invalid)

print("\nRandom samples:")
print(df.sample(10))

print("\nValidation finished.")