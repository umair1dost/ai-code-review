"""
Prediction API Routes
/predict/metrics  — Random Forest on extracted metrics
/predict/code     — LSTM + ensemble on raw source code
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    MetricsPayload, MetricsPredictionResponse,
    CodePayload, CodePredictionResponse,
    FeatureImportance,
)
from app.services.inference import get_inference_service, _severity, _build_suggestion, DECISION_THRESHOLD

router = APIRouter()


@router.post("/metrics", response_model=MetricsPredictionResponse, summary="Predict from software metrics")
async def predict_from_metrics(payload: MetricsPayload):
    """
    Accept a JSON payload of software metrics for a single code module
    and return a bug probability score from the Random Forest model,
    together with SHAP feature importance values and a review suggestion.
    """
    svc = get_inference_service()
    features = payload.model_dump()

    try:
        prob, importances = svc.predict_from_metrics(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    prediction = "BUGGY" if prob >= DECISION_THRESHOLD else "CLEAN"
    severity = _severity(prob)
    suggestion = _build_suggestion(features, [], prob)

    return MetricsPredictionResponse(
        bug_probability=round(prob, 4),
        prediction=prediction,
        severity=severity,
        top_features=[FeatureImportance(**f) for f in importances],
        review_suggestion=suggestion,
    )


@router.post("/code", response_model=CodePredictionResponse, summary="Predict from raw source code")
async def predict_from_code(payload: CodePayload):
    """
    Accept raw source code, extract metrics internally, run LSTM on
    tokenised sequences, combine via soft-voting ensemble, and return
    code smell classifications and a structured review suggestion.
    """
    svc = get_inference_service()

    try:
        rf_prob, lstm_prob, ensemble_prob, smells, extracted_features = svc.predict_from_code(
            payload.source_code, payload.language
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    prediction = "BUGGY" if ensemble_prob >= DECISION_THRESHOLD else "CLEAN"
    severity = _severity(ensemble_prob)
    suggestion = _build_suggestion(extracted_features, smells, ensemble_prob)

    return CodePredictionResponse(
        bug_probability=round(ensemble_prob, 4),
        prediction=prediction,
        severity=severity,
        code_smells=smells,
        review_suggestion=suggestion,
        ensemble_weights={"random_forest": 0.55, "lstm": 0.45},
    )
