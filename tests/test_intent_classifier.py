import pytest
from src.intent_classifier import UberIntentClassifier
from src.config import INTENTS


def test_classifier_predict_and_confidence():
    classifier = UberIntentClassifier()
    assert classifier.pipeline is not None

    intent, conf = classifier.predict("Why was my credit card overcharged for this trip?")
    assert intent == "fare_and_billing_dispute"
    assert 0.0 <= conf <= 1.0

    intent_cancel, conf_cancel = classifier.predict("I was charged a cancellation fee unfairly")
    assert intent_cancel == "cancellation_fee_issue"
    assert conf_cancel >= 0.70


def test_classifier_rule_boost():
    classifier = UberIntentClassifier()
    intent, conf = classifier.predict("I left my phone in the car")
    assert intent == "lost_item_inquiry"
    assert conf >= 0.85
