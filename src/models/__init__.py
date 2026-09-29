"""Model definitions and wrappers."""
from src.models.baseline import PhishingClassifier
from src.models.bilstm import BiLSTMClassifier

__all__ = ["PhishingClassifier", "BiLSTMClassifier"]
