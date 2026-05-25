"""
evaluate_ensemble.py
Evaluates the soft-voting ensemble (RF + LSTM) on the held-out test set
and reproduces Table 4.1 and Table 4.2 from the thesis.

Usage:
    python evaluate_ensemble.py --model_dir ../../backend/models --data_dir ../../data/processed
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RF_WEIGHT   = 0.55
LSTM_WEIGHT = 0.45
THRESHOLD   = 0.5


def evaluate(model_dir: str, data_dir: str):
    # Load models
    rf      = joblib.load(os.path.join(model_dir, "random_forest.joblib"))
    scaler  = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    import tensorflow as tf
    lstm = tf.saved_model.load(os.path.join(model_dir, "lstm_model"))

    # Load test data
    X_test_metrics = np.load(os.path.join(data_dir, "X_test_metrics.npy"))
    X_test_tokens  = np.load(os.path.join(data_dir, "X_test_tokens.npy"))
    y_test         = np.load(os.path.join(data_dir, "y_test.npy"))

    # Individual predictions
    X_scaled = scaler.transform(X_test_metrics)
    rf_proba   = rf.predict_proba(X_scaled)[:, 1]
    lstm_proba = lstm(X_test_tokens, training=False).numpy().flatten()

    # Ensemble
    ensemble_proba = RF_WEIGHT * rf_proba + LSTM_WEIGHT * lstm_proba
    ensemble_pred  = (ensemble_proba >= THRESHOLD).astype(int)
    rf_pred        = (rf_proba   >= THRESHOLD).astype(int)
    lstm_pred      = (lstm_proba >= THRESHOLD).astype(int)

    def report(name, y_true, y_pred, y_proba):
        return {
            "model":     name,
            "accuracy":  round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred), 3),
            "recall":    round(recall_score(y_true, y_pred), 3),
            "f1":        round(f1_score(y_true, y_pred), 3),
            "auc_roc":   round(roc_auc_score(y_true, y_proba), 3),
        }

    rows = [
        report("Random Forest",    y_test, rf_pred,       rf_proba),
        report("LSTM",             y_test, lstm_pred,     lstm_proba),
        report("Ensemble (Ours)",  y_test, ensemble_pred, ensemble_proba),
    ]

    print("\n=== Table 4.1: Model Performance on Held-Out Test Set ===")
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6} {'AUC-ROC':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<22} {r['accuracy']:>9.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>6.3f} {r['auc_roc']:>9.3f}")

    output_path = os.path.join(model_dir, "ensemble_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Evaluation results saved to %s", output_path)

    print("\n" + classification_report(y_test, ensemble_pred, target_names=["clean", "buggy"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../../backend/models")
    parser.add_argument("--data_dir",  default="../../data/processed")
    args = parser.parse_args()
    evaluate(args.model_dir, args.data_dir)
