def clean_text(text: str) -> str:
    """
    Clean raw text.

    Args:
        text: Raw input string.

    Returns:
        Cleaned text.
    """

    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())

    return text