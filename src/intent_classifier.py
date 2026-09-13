"""
Intent Classifier module for Uber AI Support Agent.
Implements a calibrated TF-IDF + Logistic Regression pipeline with rule-based safety boosts.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFIER_MODEL_FILE,
    INTENTS,
    INTENT_CONFIDENCE_THRESHOLD,
)
from src.preprocessing import clean_tweet_text

logger = logging.getLogger(__name__)

# Disambiguation patterns for strong signal terms
RULE_BOOSTS = {
    "cancellation_fee_issue": ["cancellation fee", "cancel fee", "charged for cancellation", "cancelled the ride", "driver cancelled"],
    "lost_item_inquiry": ["left my", "lost my", "forgot my", "left phone", "lost phone", "left wallet", "lost item", "belongings in cab"],
    "driver_behavior_and_safety": ["reckless", "drunk driver", "unprofessional driver", "harassed", "screaming driver", "refused ride", "bad attitude", "abusive"],
    "fare_and_billing_dispute": ["overcharged", "charged double", "wrong fare", "toll charge", "charged twice", "refund my money", "fare dispute"],
    "pickup_and_route_issue": ["wrong route", "took longer route", "driver went wrong way", "wrong pickup", "gps pin", "driver not moving"],
    "app_and_account_support": ["account blocked", "cant login", "phone number update", "promo code not working", "discount code", "app keeps crashing"],
}


class UberIntentClassifier:
    """
    Calibrated intent classification pipeline for customer messages.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or CLASSIFIER_MODEL_FILE
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = INTENTS
        if self.model_path.exists():
            self.load()

    def train(self, texts: List[str], labels: List[str]) -> "UberIntentClassifier":
        """
        Fits TF-IDF vectorizer and balanced Logistic Regression classifier.
        """
        cleaned_texts = [clean_tweet_text(t) for t in texts]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True,
                min_df=2,
                stop_words="english",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                C=2.0,
            )),
        ])
        self.pipeline.fit(cleaned_texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        logger.info(f"Trained intent classifier on {len(texts)} samples across {len(self.classes_)} classes.")
        return self

    def save(self, path: Optional[Path] = None):
        target = path or self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target)
        logger.info(f"Saved classifier model to {target}")

    def load(self, path: Optional[Path] = None):
        target = path or self.model_path
        if not target.exists():
            raise FileNotFoundError(f"Model file not found at {target}")
        self.pipeline = joblib.load(target)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        logger.info(f"Loaded classifier model from {target}")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Returns (predicted_intent, confidence_score).
        Applies domain-specific high-confidence keyword boosts if matched.
        """
        cleaned = clean_tweet_text(text)
        lower = cleaned.lower()

        # Check explicit rule boosts
        for intent, patterns in RULE_BOOSTS.items():
            if any(p in lower for p in patterns):
                # If rule matches, verify if model confirms or elevate rule confidence
                if self.pipeline is not None:
                    probs = self.pipeline.predict_proba([cleaned])[0]
                    idx = self.classes_.index(intent) if intent in self.classes_ else -1
                    if idx != -1 and probs[idx] > 0.25:
                        return intent, max(float(probs[idx]), 0.92)
                return intent, 0.94

        if self.pipeline is None:
            # Fallback heuristic if untrained
            return "service_feedback_and_general", 0.50

        probs = self.pipeline.predict_proba([cleaned])[0]
        max_idx = int(np.argmax(probs))
        pred_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        return pred_intent, round(confidence, 4)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Returns mapping of all intents to their respective probabilities.
        """
        cleaned = clean_tweet_text(text)
        if self.pipeline is None:
            return {intent: 1.0 / len(INTENTS) for intent in INTENTS}

        probs = self.pipeline.predict_proba([cleaned])[0]
        res = {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)}
        # Ensure all config intents exist in dict
        for intent in INTENTS:
            if intent not in res:
                res[intent] = 0.0
        return res
