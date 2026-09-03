"""
Rule-based risk detection.

This module contains simple keyword rules used to estimate
risk before applying the machine learning model.
"""

HIGH_RISK_KEYWORDS = [
    "quiero morir",
    "me quiero matar",
    "suicidio",
    "terminar con mi vida",
]

MEDIUM_RISK_KEYWORDS = [
    "ya no puedo",
    "no tiene sentido",
    "estoy cansado de vivir",
    "nadie me quiere",
]

LOW_RISK_KEYWORDS = [
    "triste",
    "solo",
    "vacío",
    "deprimido",
]

def calculate_rule_score(text: str) -> int:
    """
    Calculate a risk score based on keyword rules.

    Args:
        text: Cleaned input text.

    Returns:
        Integer risk score.
    """

    score = 0

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text:
            score += 100

    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in text:
            score += 40

    for keyword in LOW_RISK_KEYWORDS:
        if keyword in text:
            score += 10

    return score