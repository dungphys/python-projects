import os
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/physicsnet.pkl")

class ParticlePredictor:
    def __init__(self):
        self.model    = None
        self.features = None
        self.classes  = None
        self._load()

    def _load(self):
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        data          = joblib.load(MODEL_PATH)
        self.model    = data["model"]
        self.features = data["features"]
        self.classes  = data["classes"]
        print(f"Model loaded. Classes: {self.classes}")

    def predict(self, features: dict) -> dict:
        X     = np.array([[features[f] for f in self.features]])
        proba = self.model.predict_proba(X)[0]
        idx   = int(np.argmax(proba))

        return {
            "particle":      self.classes[idx],
            "confidence":    float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
            "model_version": "ensemble-v1",
        }

# Singleton
_predictor = None

def get_predictor() -> ParticlePredictor:
    global _predictor
    if _predictor is None:
        _predictor = ParticlePredictor()
    return _predictor