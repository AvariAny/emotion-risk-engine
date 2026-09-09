"""
Main analysis pipeline.

This module coordinates text preprocessing
and AI-based risk evaluation.
"""

from app.preprocessing.cleaner import clean_text
from app.services.emotion_model import predict
from app.engine.ai_rules import calculate_ai_score


def analyze_text(text: str) -> int:
    """
    Analyze text and return a risk score.
    """

    cleaned_text = clean_text(text)

    probabilities = predict(cleaned_text)

    print(probabilities)

    score = calculate_ai_score(probabilities)

    return score