from fastapi import APIRouter

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse
)

from app.services.predictor import EmotionPredictor

router = APIRouter()

predictor = EmotionPredictor()


@router.get("/")
def root():
    return {
        "message": "Emotion Risk Engine API",
        "version": "2.0.0"
    }


@router.get("/health")
def health():
    return {
        "status": "ok"
    }


@router.get("/version")
def version():
    return {
        "version": "2.0.0"
    }


@router.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    result = predictor.predict(request.text)

    return PredictionResponse(
        label=result["label"],
        class_name=result["class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
    )