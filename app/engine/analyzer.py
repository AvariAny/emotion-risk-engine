"""
Main analysis pipeline.

This module coordinates text preprocessing
and rule-based risk evaluation.
"""

from app.preprocessing.cleaner import clean_text
from app.engine.rules import calculate_rule_score


def analyze_text(text: str) -> int:
    """
    Analyze text and return a risk score.

    Args:
        text: Raw user text.

    Returns:
        Integer risk score.
    """

    cleaned_text = clean_text(text)

    score = calculate_rule_score(cleaned_text)

    return score