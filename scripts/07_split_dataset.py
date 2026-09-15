from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/emotion_risk_dataset.csv")
OUTPUT = Path("data/processed")

df = pd.read_csv(INPUT)

# 80% train, 20% temporal
train, temp = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)

# 10% validation, 10% test
validation, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42,
)

train.to_csv(OUTPUT / "train.csv", index=False)
validation.to_csv(OUTPUT / "validation.csv", index=False)
test.to_csv(OUTPUT / "test.csv", index=False)

print("=" * 60)
print("SPLIT COMPLETADO")
print("=" * 60)
print(f"Train      : {len(train)}")
print(f"Validation : {len(validation)}")
print(f"Test       : {len(test)}")
print(f"Total      : {len(df)}")