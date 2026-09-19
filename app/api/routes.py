from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Prediction

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistory,
    PredictionUpdate
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
def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):

    result = predictor.predict(
        request.text,
        db
    )

    return PredictionResponse(
        label=result["label"],
        class_name=result["class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"]
    )


# ============================================================
# History
# ============================================================

@router.get(
    "/history",
    response_model=list[PredictionHistory]
)
def history(
    db: Session = Depends(get_db)
):
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return predictions


@router.get(
    "/history/{prediction_id}",
    response_model=PredictionHistory
)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    return prediction


# ============================================================
# Update Prediction
# ============================================================

@router.patch(
    "/history/{prediction_id}",
    response_model=PredictionHistory
)
def update_prediction_patch(
    prediction_id: int,
    payload: PredictionUpdate,
    db: Session = Depends(get_db)
):

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    prediction.risk = payload.risk

    db.commit()
    db.refresh(prediction)

    return prediction


@router.put(
    "/history/{prediction_id}",
    response_model=PredictionHistory
)
def update_prediction_put(
    prediction_id: int,
    request: PredictionUpdate,
    db: Session = Depends(get_db)
):

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    prediction.risk = request.risk

    db.commit()
    db.refresh(prediction)

    return prediction


# ============================================================
# Delete Prediction
# ============================================================

@router.delete(
    "/history/{prediction_id}",
    status_code=204
)
def delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    db.delete(prediction)
    db.commit()

    return