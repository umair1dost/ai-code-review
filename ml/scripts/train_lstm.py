"""
train_lstm.py
Trains the LSTM network on tokenised source code sequences.

Usage:
    python train_lstm.py --config ../../configs/lstm_config.yaml
"""

import argparse
import json
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_lstm_model(vocab_size: int = 10_000, embed_dim: int = 128,
                     seq_len: int = 512, dropout: float = 0.3):
    """
    Architecture from the thesis:
        Embedding(vocab_size, embed_dim)
        → LSTM(128, return_sequences=True) + Dropout
        → LSTM(64)                         + Dropout
        → Dense(64, relu)
        → Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(seq_len,), name="token_ids")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)
    x = layers.LSTM(128, return_sequences=True, name="lstm_1")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.LSTM(64, name="lstm_2")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="lstm_bug_predictor")
    return model


def train(config: dict, data_dir: str, output_dir: str):
    import tensorflow as tf

    os.makedirs(output_dir, exist_ok=True)

    # Load pre-tokenised numpy arrays
    X_train = np.load(os.path.join(data_dir, "X_train_tokens.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val_tokens.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test_tokens.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    logger.info("Data loaded: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    vocab_size = config.get("vocab_size", 10_000)
    embed_dim  = config.get("embed_dim", 128)
    seq_len    = config.get("seq_len", 512)
    dropout    = config.get("dropout", 0.3)
    epochs     = config.get("epochs", 30)
    batch_size = config.get("batch_size", 64)
    lr         = config.get("learning_rate", 0.001)
    patience   = config.get("early_stopping_patience", 5)

    model = build_lstm_model(vocab_size, embed_dim, seq_len, dropout)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc_roc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "lstm_best.h5"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "logs"), histogram_freq=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, [round(float(v), 4) for v in results]))
    logger.info("Test set results: %s", metrics)

    # Save as TensorFlow SavedModel (used by inference service)
    saved_model_path = os.path.join(output_dir, "lstm_model")
    model.save(saved_model_path)
    logger.info("SavedModel saved to %s", saved_model_path)

    with open(os.path.join(output_dir, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="../../configs/lstm_config.yaml")
    parser.add_argument("--data_dir", default="../../data/processed")
    parser.add_argument("--output",   default="../../backend/models")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    train(cfg, args.data_dir, args.output)
