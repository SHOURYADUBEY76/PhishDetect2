"""Character-level Bidirectional LSTM (BiLSTM) Classifier.
Implements deep learning benchmark trained on character sequence representations of raw URLs.
Guarantees identical data partitions for a fair comparison against tabular tree models.
"""

import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from src.utils.metrics import find_optimal_threshold, calculate_metrics, measure_inference_latency


class CharacterTokenizer:
    """Character-level vocabulary tokenizer for raw URL strings."""

    def __init__(self, max_length: int = 150):
        self.max_length = max_length
        # Printable ASCII characters vocabulary (indices 1..95, UNK=96, padding=0)
        chars = [chr(i) for i in range(32, 127)]
        self.char_to_idx = {c: idx + 1 for idx, c in enumerate(chars)}
        self.char_to_idx["<UNK>"] = len(self.char_to_idx) + 1
        self.vocab_size = len(self.char_to_idx) + 1

    def texts_to_sequences(self, urls: list[str]) -> np.ndarray:
        """Convert list of URL strings to 2D integer sequence array."""
        sequences = []
        unk_idx = self.char_to_idx["<UNK>"]
        for u in urls:
            u_str = str(u).strip()
            seq = [self.char_to_idx.get(c, unk_idx) for c in u_str]
            sequences.append(seq)
        return pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")


class BiLSTMClassifier:
    """BiLSTM neural network classifier for URL strings."""

    def __init__(self, max_length: int = 150, embedding_dim: int = 32, lstm_units: int = 64, random_state: int = 42):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.random_state = random_state
        self.tokenizer = CharacterTokenizer(max_length=max_length)
        self.optimal_threshold = 0.5
        self.model = None

    def _build_model(self):
        tf.random.set_seed(self.random_state)
        model = Sequential([
            Embedding(
                input_dim=self.tokenizer.vocab_size,
                output_dim=self.embedding_dim
            ),
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, train_urls: list[str], y_train: np.ndarray, val_urls: list[str] = None, y_val: np.ndarray = None, epochs: int = 8, batch_size: int = 128):
        """Train BiLSTM network with early stopping."""
        X_tr = self.tokenizer.texts_to_sequences(train_urls)
        self.model = self._build_model()

        validation_data = None
        callbacks = []

        if val_urls is not None and y_val is not None and len(y_val) > 0:
            X_vl = self.tokenizer.texts_to_sequences(val_urls)
            validation_data = (X_vl, y_val)
            callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

        self.model.fit(
            X_tr, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Calibrate threshold on validation data
        if val_urls is not None and y_val is not None and len(y_val) > 0:
            val_probs = self.predict_proba_urls(val_urls)[:, 1]
            self.optimal_threshold = find_optimal_threshold(y_val, val_probs)
        else:
            self.optimal_threshold = 0.5

        return self

    def predict_proba_urls(self, urls: list[str]) -> np.ndarray:
        """Predict probabilities from URL strings."""
        X = self.tokenizer.texts_to_sequences(urls)
        pos_probs = self.model.predict(X, verbose=0).flatten()
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict probabilities from pre-tokenized sequence array."""
        pos_probs = self.model.predict(X_seq, verbose=0).flatten()
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def evaluate_urls(self, test_urls: list[str], y_test: np.ndarray) -> dict:
        """Evaluate BiLSTM on URL list and return metrics."""
        probs = self.predict_proba_urls(test_urls)[:, 1]
        X_sample = self.tokenizer.texts_to_sequences(test_urls[:100])
        latency = measure_inference_latency(self, X_sample)

        # Approximate model size from param count
        size_mb = (self.model.count_params() * 4) / (1024 * 1024) if self.model else 0.0

        return calculate_metrics(
            y_true=y_test,
            y_probs=probs,
            threshold=self.optimal_threshold,
            latency_ms=latency,
            model_size_mb=float(size_mb)
        )
