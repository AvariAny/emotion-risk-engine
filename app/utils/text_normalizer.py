import re

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return text

    text = text[0].upper() + text[1:]

    if text[-1] not in ".!?":
        text += "."

    return text