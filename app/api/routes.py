import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.database import get_db
from app.database.models import Prediction

from app.utils.config import LABELS
from app.utils.logger import logger

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistory,
    PredictionUpdate,
    PredictionStats,
    AlertResponse
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

    logger.info(
        f'Prediction | text="{request.text}" | '
        f'risk={result["label"]} | '
        f'confidence={result["confidence"]:.4f}'
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
        logger.warning(
            f"Prediction {prediction_id} not found"
        )
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
        logger.warning(
            f"Prediction {prediction_id} not found for PATCH"
        )
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    prediction.risk = payload.risk

    db.commit()
    db.refresh(prediction)

    logger.info(
        f"Prediction {prediction_id} updated (PATCH) "
        f"risk={payload.risk}"
    )

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
        logger.warning(
            f"Prediction {prediction_id} not found for PUT"
        )
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    prediction.risk = request.risk

    db.commit()
    db.refresh(prediction)

    logger.info(
        f"Prediction {prediction_id} updated (PUT) "
        f"risk={request.risk}"
    )

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
        logger.warning(
            f"Prediction {prediction_id} not found for DELETE"
        )
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    db.delete(prediction)
    db.commit()

    logger.info(f"Prediction {prediction_id} deleted")

    return


# ============================================================
# Stats
# ============================================================

@router.get(
    "/stats",
    response_model=PredictionStats
)
def stats(
    db: Session = Depends(get_db)
):

    total = db.query(Prediction).count()

    avg_confidence = (
        db.query(func.avg(Prediction.confidence))
        .scalar()
    ) or 0.0

    distribution_rows = (
        db.query(Prediction.risk, func.count(Prediction.risk))
        .group_by(Prediction.risk)
        .all()
    )

    # Inicializamos TODAS las clases posibles a 0,
    # usando los índices de LABELS para no hardcodear.
    risk_distribution = {
        i: 0 for i in range(len(LABELS))
    }

    # Sobrescribimos solo las clases que sí aparecen en la BD.
    for risk, count in distribution_rows:
        risk_distribution[risk] = count

    return PredictionStats(
        total_predictions=total,
        average_confidence=round(float(avg_confidence), 4),
        risk_distribution=risk_distribution
    )


# ============================================================
# Alerts
# ============================================================

@router.get(
    "/alerts",
    response_model=list[AlertResponse]
)
def alerts(
    db: Session = Depends(get_db),
    risk_threshold: int = 3
):
    return (
        db.query(Prediction)
        .filter(Prediction.risk >= risk_threshold)
        .order_by(Prediction.created_at.desc())
        .all()
    )


# ============================================================
# Export JSON
# ============================================================

@router.get("/export/json")
def export_json(
    db: Session = Depends(get_db)
):

    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    data = []

    for prediction in predictions:

        data.append(
            {
                "id": prediction.id,
                "text": prediction.text,
                "risk": prediction.risk,
                "confidence": prediction.confidence,
                "created_at": prediction.created_at.isoformat()
            }
        )

    logger.info(
        f"Exported {len(data)} predictions to JSON"
    )

    return JSONResponse(content=data)


# ============================================================
# Export CSV
# ============================================================

@router.get("/export/csv")
def export_csv(
    db: Session = Depends(get_db)
):

    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "id",
        "text",
        "risk",
        "confidence",
        "created_at"
    ])

    for prediction in predictions:
        writer.writerow([
            prediction.id,
            prediction.text,
            prediction.risk,
            prediction.confidence,
            prediction.created_at
        ])

    output.seek(0)

    logger.info(
        f"Exported {len(predictions)} predictions to CSV"
    )

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=predictions.csv"
        }
    )