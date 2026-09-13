"""
Escalation and safety triage engine for Uber AI Support Agent.
Implements calibrated rules and confidence thresholds to determine whether
an inquiry can be reliably auto-handled or must be escalated to a human specialist.
"""
from typing import Dict, List, Optional, Tuple

from src.config import (
    ACCOUNT_SECURITY_KEYWORDS,
    CRITICAL_SAFETY_KEYWORDS,
    DECISION_AUTO_HANDLE,
    DECISION_ESCALATE,
    INTENT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
)
from src.preprocessing import clean_tweet_text


def evaluate_escalation(
    customer_message: str,
    intent: str,
    intent_confidence: float,
    evidence: List[Dict],
    confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
    similarity_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD,
) -> Dict:
    """
    Evaluates customer inquiry, intent confidence, and retrieved historical evidence
    to output an escalation decision and auditable rationale.
    """
    lower_msg = clean_tweet_text(customer_message).lower()

    # Rule 1: Critical Safety & Misconduct Trigger (Immediate Human Escalation)
    for kw in CRITICAL_SAFETY_KEYWORDS:
        if kw in lower_msg:
            return {
                "decision": DECISION_ESCALATE,
                "escalation_reason": f"Critical safety trigger detected (keyword: '{kw}'). Driver misconduct and safety emergencies require human triage.",
                "rule_triggered": "SAFETY_HAZARD_OVERRIDE",
                "is_safety_escalation": True,
            }

    # Rule 2: Account Security & Fraud Trigger
    for kw in ACCOUNT_SECURITY_KEYWORDS:
        if kw in lower_msg:
            return {
                "decision": DECISION_ESCALATE,
                "escalation_reason": f"Account security/fraud indicator detected (keyword: '{kw}'). Requires verified account identity review.",
                "rule_triggered": "ACCOUNT_SECURITY_OVERRIDE",
                "is_safety_escalation": False,
            }

    # Rule 3: Intent Classification Confidence Check
    if intent_confidence < confidence_threshold:
        return {
            "decision": DECISION_ESCALATE,
            "escalation_reason": f"Low intent confidence ({intent_confidence:.2f} < {confidence_threshold:.2f}). Query is ambiguous or multi-faceted.",
            "rule_triggered": "LOW_INTENT_CONFIDENCE",
            "is_safety_escalation": False,
        }

    # Rule 4: Historical Evidence Sufficiency Check
    if not evidence:
        return {
            "decision": DECISION_ESCALATE,
            "escalation_reason": "No relevant historical support resolution found in knowledge base.",
            "rule_triggered": "MISSING_HISTORICAL_EVIDENCE",
            "is_safety_escalation": False,
        }

    top_evidence = evidence[0]
    top_similarity = top_evidence.get("similarity", 0.0)

    if top_similarity < similarity_threshold:
        return {
            "decision": DECISION_ESCALATE,
            "escalation_reason": f"Historical evidence similarity ({top_similarity:.2f}) is below groundedness threshold ({similarity_threshold:.2f}).",
            "rule_triggered": "LOW_EVIDENCE_SIMILARITY",
            "is_safety_escalation": False,
        }

    # Rule 5: Driver Behavior Intent (non-critical, but driver disputes still require human review)
    if intent == "driver_behavior_and_safety":
        return {
            "decision": DECISION_ESCALATE,
            "escalation_reason": "Driver behavior dispute requires supervisor investigation and driver history check.",
            "rule_triggered": "DRIVER_CONDUCT_REVIEW",
            "is_safety_escalation": False,
        }

    # Auto-handle: Meets all confidence and evidence thresholds
    return {
        "decision": DECISION_AUTO_HANDLE,
        "escalation_reason": None,
        "rule_triggered": "AUTO_HANDLE_QUALIFIED",
        "is_safety_escalation": False,
    }
