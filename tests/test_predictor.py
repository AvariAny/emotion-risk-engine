from app.services.predictor import EmotionPredictor

predictor = EmotionPredictor()

resultado = predictor.predict(
    "No quiero seguir viviendo."
)

print(resultado)