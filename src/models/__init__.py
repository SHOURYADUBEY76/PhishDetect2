"""Model definitions and wrappers.
BiLSTMClassifier is lazily imported to prevent eager TensorFlow loading on API/CLI startup.
"""
from src.models.baseline import PhishingClassifier


def __getattr__(name: str):
    if name == "BiLSTMClassifier":
        from src.models.bilstm import BiLSTMClassifier
        return BiLSTMClassifier
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["PhishingClassifier", "BiLSTMClassifier"]
