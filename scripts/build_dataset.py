import pandas as pd

FILES = [
    "data/raw/label0_claude_01.csv",
    "data/raw/label0_deepseek_01.csv",
    "data/raw/label1_claude_01.csv",
    "data/raw/label1_deepseek_01.csv",
    "data/raw/label2_claude_01.csv",
    "data/raw/label2_deepseek_01.csv",
    "data/raw/label3_deepseek_01.csv",
]

dfs = []

for file in FILES:
    try:
        df = pd.read_csv(file)
        # Asegurar que las columnas existan
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"Advertencia: {file} no contiene las columnas 'text' y 'label'. Se omite.")
            continue
        # Convertir label a entero (elimina espacios, comillas, etc.)
        df['label'] = df['label'].astype(str).str.strip().astype(int)
        dfs.append(df)
    except Exception as e:
        print(f"Error al leer {file}: {e}")

if not dfs:
    print("No se cargaron datos. Saliendo.")
    exit()

dataset = pd.concat(dfs, ignore_index=True)

# Eliminar duplicados por texto
dataset = dataset.drop_duplicates(subset=['text'])

# Mezclar aleatoriamente
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
dataset.to_csv("data/processed/emotion_risk_dataset.csv", index=False)

print("Dataset generado correctamente.")
print(f"Total ejemplos: {len(dataset)}")
print("\nDistribución de etiquetas:")
print(dataset['label'].value_counts().sort_index())