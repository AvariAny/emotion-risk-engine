import logging

logger = logging.getLogger("emotion-risk-engine")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

# Consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Archivo
file_handler = logging.FileHandler(
    "emotion_risk_engine.log"
)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)