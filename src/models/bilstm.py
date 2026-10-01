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

    def __init__(
        self,
        max_length: int = 150,
        embedding_dim: int = 32,
        lstm_units: int = 64,
        optimizer_type: str = "lion",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        random_state: int = 42
    ):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.tokenizer = CharacterTokenizer(max_length=max_length)
        self.optimal_threshold = 0.5
        self.model = None
        self.history = None

    def _build_optimizer(self):
        """Build specified optimizer (Lion or Adam) with tuned hyperparams."""
        if self.optimizer_type == "lion":
            from tensorflow.keras.optimizers import Lion
            return Lion(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adam":
            from tensorflow.keras.optimizers import Adam
            return Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer_type '{self.optimizer_type}'. Choose 'lion' or 'adam'.")

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
        optimizer = self._build_optimizer()
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        return model

    def fit(self, train_urls: list[str], y_train: np.ndarray, val_urls: list[str] = None, y_val: np.ndarray = None, epochs: int = 8, batch_size: int = 256, verbose: int = 1):
        """Train BiLSTM network with early stopping and history logging."""
        X_tr = self.tokenizer.texts_to_sequences(train_urls)
        self.model = self._build_model()

        validation_data = None
        callbacks = []

        if val_urls is not None and y_val is not None and len(y_val) > 0:
            X_vl = self.tokenizer.texts_to_sequences(val_urls)
            validation_data = (X_vl, y_val)
            callbacks = [EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]

        fit_res = self.model.fit(
            X_tr, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        self.history = fit_res.history

        # Calibrate threshold on validation data
        if val_urls is not None and y_val is not None and len(y_val) > 0:
            val_probs = self.predict_proba_urls(val_urls)[:, 1]
            self.optimal_threshold = find_optimal_threshold(y_val, val_probs)
        else:
            self.optimal_threshold = 0.5

        return self

    def predict_proba_urls(self, urls: list[str], batch_size: int = 512) -> np.ndarray:
        """Predict probabilities from URL strings."""
        X = self.tokenizer.texts_to_sequences(urls)
        pos_probs = self.model.predict(X, batch_size=batch_size, verbose=0).flatten()
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def predict_proba(self, X_seq: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict probabilities from pre-tokenized sequence array."""
        pos_probs = self.model.predict(X_seq, batch_size=batch_size, verbose=0).flatten()
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

    def save(self, filepath: str):
        """Save model weights, configuration, and tokenizer."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Save keras model
        model_path = filepath if filepath.endswith(".keras") else f"{filepath}.keras"
        self.model.save(model_path)
        # Save metadata and tokenizer
        meta_path = f"{filepath}_meta.joblib" if not filepath.endswith(".keras") else filepath.replace(".keras", "_meta.joblib")
        meta = {
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "optimal_threshold": self.optimal_threshold,
            "char_to_idx": self.tokenizer.char_to_idx,
            "vocab_size": self.tokenizer.vocab_size,
            "history": self.history
        }
        joblib.dump(meta, meta_path)

    @classmethod
    def load(cls, filepath: str):
        """Load trained BiLSTM model and tokenizer from saved artifacts."""
        model_path = filepath if filepath.endswith(".keras") else f"{filepath}.keras"
        meta_path = f"{filepath}_meta.joblib" if not filepath.endswith(".keras") else filepath.replace(".keras", "_meta.joblib")

        meta = joblib.load(meta_path)
        clf = cls(
            max_length=meta["max_length"],
            embedding_dim=meta["embedding_dim"],
            lstm_units=meta["lstm_units"],
            optimizer_type=meta["optimizer_type"],
            learning_rate=meta["learning_rate"],
            weight_decay=meta["weight_decay"],
            random_state=meta["random_state"]
        )
        clf.optimal_threshold = meta["optimal_threshold"]
        clf.tokenizer.char_to_idx = meta["char_to_idx"]
        clf.tokenizer.vocab_size = meta["vocab_size"]
        clf.history = meta.get("history")
        clf.model = load_model(model_path)
        return clf

