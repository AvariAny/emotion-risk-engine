from pathlib import Path

RAW_PATH = Path("data/raw")

LABELS = [0, 1, 2, 3]

TOPICS = [
    "work",
    "school",
    "university",
    "programming",
    "gaming",
    "sports",
    "pets",
    "technology",
    "shopping",
    "transportation",
    "travel",
    "movies",
    "music",
    "books",
    "hobbies",
    "family",
    "friends",
    "cooking",
    "weather",
    "restaurants",
]

MODELS = [
    "claude",
    "gemini",
]

for label in LABELS:

    for topic in TOPICS:

        folder = RAW_PATH / f"label{label}" / topic
        folder.mkdir(parents=True, exist_ok=True)

        for model in MODELS:

            file = folder / f"{model}.csv"

            if not file.exists():
                file.write_text(
                    "text,label\n",
                    encoding="utf-8"
                )

print("✅ Estructura creada correctamente.")