"""
Trivial Baseline for Uber AI Support Agent.
Implements a simple majority-class intent predictor with a static canned response.
Acts as the lower-bound benchmark for evaluation.
"""
from typing import Dict, List, Optional
from src.config import BRAND_NAME, DECISION_AUTO_HANDLE


class TrivialBaselineAgent:
    """
    Trivial baseline: predicts majority intent and outputs a fixed generic canned template.
    """

    def __init__(self, majority_intent: str = "fare_and_billing_dispute"):
        self.majority_intent = majority_intent

    def process_message(self, customer_message: str, **kwargs) -> Dict:
        canned_reply = (
            f"Hi there! Thank you for contacting {BRAND_NAME} Support. "
            "Please visit help.uber.com or check the 'Help' section in your Uber app for assistance."
        )
        return {
            "brand": BRAND_NAME,
            "customer_message": customer_message,
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "reply": canned_reply,
            "decision": DECISION_AUTO_HANDLE,
            "escalation_reason": None,
            "evidence": [],
            "baseline_type": "trivial_majority_class",
        }
