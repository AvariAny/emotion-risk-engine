"""
Risk scoring based on emotion probabilities.
"""

ANGER = 2
FEAR = 14
GRIEF = 16
REMORSE = 25
SADNESS = 26


def calculate_ai_score(probabilities: list[float]) -> int:
    """
    Convert emotion probabilities into a 0–100 risk score.
    """

    score = (
        probabilities[SADNESS] * 40 +
        probabilities[GRIEF] * 25 +
        probabilities[FEAR] * 15 +
        probabilities[REMORSE] * 10 +
        probabilities[ANGER] * 10
    )

    return min(100, round(score * 100))