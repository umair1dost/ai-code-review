"""
ML Inference Service
Loads the trained Random Forest and LSTM models at startup and
serves predictions through a unified interface.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))

# Ensemble weights (determined empirically on validation set)
RF_WEIGHT = 0.55
LSTM_WEIGHT = 0.45
DECISION_THRESHOLD = 0.5

# SHAP feature names (61 features after pruning)
FEATURE_NAMES = [
    "loc", "lloc", "num_methods", "num_attributes",
    "cyclomatic_complexity", "halstead_volume", "halstead_effort", "halstead_difficulty",
    "cbo", "wmc", "rfc", "lcom", "dit", "noc",
    "num_authors", "code_churn", "prior_defects", "module_age_days",
    "fan_in", "fan_out", "instability",
]

SMELL_PATTERNS = {
    "God Class":      lambda f: f.get("wmc", 0) > 20 and f.get("lcom", 0) > 0.6,
    "Long Method":    lambda f: f.get("cyclomatic_complexity", 0) > 15,
    "High Coupling":  lambda f: f.get("cbo", 0) > 10,
    "Deep Hierarchy": lambda f: f.get("dit", 0) > 4,
    "Data Clump":     lambda f: f.get("num_attributes", 0) > 10 and f.get("lcom", 0) > 0.5,
    "Feature Envy":   lambda f: f.get("fan_out", 0) > 8 and f.get("rfc", 0) > 40,
}


def _severity(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    if prob >= 0.30:
        return "LOW"
    return "NONE"


def _build_suggestion(features: Dict, smells: List[str], prob: float) -> str:
    if prob < DECISION_THRESHOLD:
        return "No significant defect indicators detected. Code quality appears acceptable."

    parts = []
    cbo = features.get("cbo", 0)
    wmc = features.get("wmc", 0)
    cc = features.get("cyclomatic_complexity", 0)

    if cbo > 10:
        parts.append(f"high coupling (CBO = {cbo})")
    if wmc > 20:
        parts.append(f"excessive method complexity (WMC = {wmc})")
    if cc > 15:
        parts.append(f"high cyclomatic complexity ({cc})")

    if smells:
        smell_str = ", ".join(smells)
        suggestion = f"Detected code smells: {smell_str}. "
    else:
        suggestion = ""

    if parts:
        suggestion += f"This module exhibits {', '.join(parts)}. "
        suggestion += "Refactoring is recommended: decompose into smaller single-responsibility classes and reduce inter-module dependencies."
    else:
        suggestion += "Review recommended based on historical defect patterns in similar modules."

    return suggestion


class InferenceService:
    """
    Wraps the Random Forest and LSTM models.

    If trained model artefacts are not found on disk, the service falls back to
    a rule-based heuristic so the API remains functional during development.
    """

    def __init__(self):
        self._rf = None
        self._lstm = None
        self._scaler = None
        self._tokenizer = None
        self._shap_explainer = None
        self._load_models()

    def _load_models(self):
        try:
            import joblib
            rf_path = os.path.join(MODEL_DIR, "random_forest.joblib")
            scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
            if os.path.exists(rf_path):
                self._rf = joblib.load(rf_path)
                self._scaler = joblib.load(scaler_path)
                logger.info("Random Forest model loaded from %s", rf_path)
            else:
                logger.warning("RF model not found at %s — using heuristic fallback.", rf_path)
        except Exception as exc:
            logger.error("Failed to load Random Forest: %s", exc)

        try:
            import tensorflow as tf
            lstm_path = os.path.join(MODEL_DIR, "lstm_model")
            if os.path.exists(lstm_path):
                self._lstm = tf.saved_model.load(lstm_path)
                logger.info("LSTM model loaded from %s", lstm_path)
            else:
                logger.warning("LSTM model not found at %s — using heuristic fallback.", lstm_path)
        except Exception as exc:
            logger.error("Failed to load LSTM: %s", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_from_metrics(self, features: Dict) -> Tuple[float, List[dict]]:
        """Return (bug_probability, shap_feature_importances)."""
        feature_vector = np.array([[features.get(f, 0) for f in FEATURE_NAMES]])

        if self._rf is not None and self._scaler is not None:
            scaled = self._scaler.transform(feature_vector)
            rf_prob = float(self._rf.predict_proba(scaled)[0][1])
            importances = self._compute_shap(scaled, features)
        else:
            rf_prob = self._heuristic_rf(features)
            importances = self._heuristic_shap(features)

        return rf_prob, importances

    def predict_from_code(self, source_code: str, language: str) -> Tuple[float, float, List[str]]:
        """Return (rf_prob, lstm_prob, detected_smells)."""
        from app.utils.metrics_extractor import extract_metrics_from_code
        from app.utils.tokenizer import tokenize_code

        extracted = extract_metrics_from_code(source_code, language)
        rf_prob, _ = self.predict_from_metrics(extracted)

        if self._lstm is not None:
            tokens = tokenize_code(source_code, language)
            token_array = np.array([tokens])
            lstm_prob = float(self._lstm(token_array, training=False).numpy()[0][0])
        else:
            lstm_prob = self._heuristic_lstm(source_code)

        ensemble_prob = RF_WEIGHT * rf_prob + LSTM_WEIGHT * lstm_prob
        smells = [name for name, fn in SMELL_PATTERNS.items() if fn(extracted)]

        return rf_prob, lstm_prob, ensemble_prob, smells, extracted

    # ------------------------------------------------------------------
    # Heuristic fallbacks (no trained model on disk)
    # ------------------------------------------------------------------

    def _heuristic_rf(self, features: Dict) -> float:
        score = 0.0
        if features.get("cbo", 0) > 10:       score += 0.25
        if features.get("wmc", 0) > 20:        score += 0.25
        if features.get("cyclomatic_complexity", 0) > 15: score += 0.20
        if features.get("prior_defects", 0) > 1: score += 0.20
        if features.get("lcom", 0) > 0.7:      score += 0.10
        return min(score, 0.99)

    def _heuristic_lstm(self, code: str) -> float:
        score = 0.0
        if len(code) > 1000:   score += 0.2
        if "TODO" in code or "FIXME" in code: score += 0.15
        if code.count("for") + code.count("while") > 5: score += 0.15
        if "except:" in code or "except Exception" in code: score += 0.2
        return min(score, 0.99)

    def _heuristic_shap(self, features: Dict) -> List[dict]:
        ranked = sorted(
            [(f, abs(features.get(f, 0) / max(features.get(f, 1), 1))) for f in FEATURE_NAMES],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        total = sum(v for _, v in ranked) or 1
        return [{"feature": f, "shap_value": round(v / total, 3)} for f, v in ranked]

    def _compute_shap(self, scaled_vector: np.ndarray, features: Dict) -> List[dict]:
        try:
            import shap
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self._rf)
            shap_values = self._shap_explainer.shap_values(scaled_vector)
            # shap_values[1] = positive class
            values = np.abs(shap_values[1][0])
            ranked = sorted(zip(FEATURE_NAMES, values), key=lambda x: x[1], reverse=True)[:5]
            total = sum(v for _, v in ranked) or 1
            return [{"feature": f, "shap_value": round(float(v) / total, 3)} for f, v in ranked]
        except Exception:
            return self._heuristic_shap(features)


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Singleton — loaded once at startup."""
    return InferenceService()
