from fastapi import FastAPI
from pydantic import BaseModel

from app.services.emotion_model import EmotionRiskModel

app = FastAPI(
    title="Emotion Risk Engine API",
    version="1.0.0",
    description="API para evaluar riesgo emocional utilizando DistilBERT.",
)

model = EmotionRiskModel()


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Emotion Risk Engine API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/version")
def version():
    return {
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    return model.predict(request.text)