"""
train_random_forest.py
Trains the Random Forest classifier on the extracted software metrics dataset.

Usage:
    python train_random_forest.py --config ../../configs/rf_config.yaml
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    val_path   = os.path.join(data_dir, "val.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)

    logger.info("Loaded %d train / %d val / %d test samples", len(train), len(val), len(test))
    return train, val, test


def prepare_features(df: pd.DataFrame, target_col: str = "label"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info("Dropping %d correlated features: %s", len(to_drop), to_drop)
    return X.drop(columns=to_drop)


def train(config: dict, data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, test_df = load_data(data_dir)

    X_train, y_train = prepare_features(train_df)
    X_val,   y_val   = prepare_features(val_df)
    X_test,  y_test  = prepare_features(test_df)

    # Feature selection
    X_train = remove_correlated_features(X_train, threshold=config.get("corr_threshold", 0.9))
    feature_cols = X_train.columns.tolist()
    X_val  = X_val[feature_cols]
    X_test = X_test[feature_cols]

    logger.info("Training with %d features", len(feature_cols))

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # SMOTE — training split only
    sm = SMOTE(random_state=42)
    X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
    logger.info("After SMOTE: %d samples (%.1f%% buggy)",
                len(y_train), 100 * y_train.mean())

    # Grid search
    param_grid = config.get("param_grid", {
        "n_estimators": [100, 200, 500],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", None],
    })

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="f1",
        verbose=2,
        n_jobs=-1,
    )
    logger.info("Starting grid search ...")
    gs.fit(X_train_s, y_train)
    logger.info("Best params: %s  |  CV F1: %.4f", gs.best_params_, gs.best_score_)

    best_rf = gs.best_estimator_

    # Evaluate on held-out test set
    y_pred  = best_rf.predict(X_test_s)
    y_proba = best_rf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info("Test set results: %s", metrics)
    print(classification_report(y_test, y_pred, target_names=["clean", "buggy"]))

    # Save artefacts
    joblib.dump(best_rf, os.path.join(output_dir, "random_forest.joblib"))
    joblib.dump(scaler,  os.path.join(output_dir, "scaler.joblib"))
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "rf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artefacts saved to %s", output_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="../../configs/rf_config.yaml")
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--output",   default="../../backend/models")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    train(cfg, args.data_dir, args.output)
