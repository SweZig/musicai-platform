"""
Genre-klassificerare — Random Forest baseline.
Fas 1: Regel-baserad klassificering från BPM + spektrala features.
Fas 2: Ersätts med tränad RF-modell på GTZAN/FMA-data.
"""
import os
import pickle
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger()

GENRES = ["electronic", "rock", "pop", "hip-hop", "jazz", "classical", "country", "unknown"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../ml/models/genre_classifier.pkl")


class GenreClassifier:

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """Ladda tränad modell om den finns, annars använd regelbaserad fallback."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                log.info("classifier_model_loaded", path=MODEL_PATH)
                return model
            except Exception as e:
                log.warning("classifier_load_failed", error=str(e))
        log.info("classifier_using_heuristic_fallback")
        return None

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Klassificera genre baserat på features."""
        feature_vector = features.get("feature_vector")

        if self.model is not None and feature_vector:
            return self._predict_ml(feature_vector)
        else:
            return self._predict_heuristic(features)

    def _predict_ml(self, feature_vector: list) -> dict:
        """Prediktera med tränad modell."""
        X = np.array(feature_vector).reshape(1, -1)
        genre = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        confidence = round(float(proba.max()), 4)
        return {
            "genre": genre,
            "subgenre": None,
            "confidence": confidence,
            "scores": scores,
            "method": "random_forest",
        }

    def _predict_heuristic(self, features: dict) -> dict:
        """
        Regelbaserad heuristik — används tills modellen är tränad.
        Grova regler baserade på BPM och spektrala egenskaper.
        """
        bpm = features.get("bpm", 0)
        zcr = features.get("zero_crossing_rate_mean", 0)
        energy = features.get("energy", 0)
        spectral_centroid = features.get("spectral_centroid_mean", 0)

        scores = {g: 0.0 for g in GENRES}

        # Electronic: 115-145 BPM, hög energi
        if 115 <= bpm <= 145:
            scores["electronic"] += 0.5
        if energy > 0.01:
            scores["electronic"] += 0.2

        # Rock: 100-160 BPM, hög ZCR (distorsion), hög spektral centroid
        if 100 <= bpm <= 160 and zcr > 0.05:
            scores["rock"] += 0.5
        if spectral_centroid > 3000:
            scores["rock"] += 0.2

        # Hip-Hop: 70-100 BPM, låg spektral centroid
        if 70 <= bpm <= 100:
            scores["hip-hop"] += 0.4
        if spectral_centroid < 2000:
            scores["hip-hop"] += 0.2

        # Jazz: 80-200 BPM, hög harmonisk komplexitet (chroma-variation)
        chroma_std = features.get("chroma_stats", {}).get("std", [0])
        if isinstance(chroma_std, list) and len(chroma_std) > 0:
            if np.mean(chroma_std) > 0.1:
                scores["jazz"] += 0.4

        # Classical: låg ZCR, varierande energi
        if zcr < 0.02:
            scores["classical"] += 0.3

        # Pop: 100-130 BPM, medel energi
        if 100 <= bpm <= 130 and 0.005 < energy < 0.05:
            scores["pop"] += 0.4

        # Normalisera scores
        total = sum(scores.values()) or 1
        scores = {k: round(v / total, 4) for k, v in scores.items()}

        best_genre = max(scores, key=scores.get)
        confidence = scores[best_genre]

        if confidence < 0.3:
            best_genre = "unknown"

        return {
            "genre": best_genre,
            "subgenre": None,
            "confidence": confidence,
            "scores": scores,
            "method": "heuristic",
        }
