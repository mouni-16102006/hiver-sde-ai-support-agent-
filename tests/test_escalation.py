import pytest
from src.escalation import evaluate_escalation
from src.config import DECISION_AUTO_HANDLE, DECISION_ESCALATE


def test_safety_override_escalation():
    msg = "Driver was driving 90mph, drunk and weaving recklessly on the highway!"
    res = evaluate_escalation(
        customer_message=msg,
        intent="driver_behavior_and_safety",
        intent_confidence=0.95,
        evidence=[{"similarity": 0.85}],
    )
    assert res["decision"] == DECISION_ESCALATE
    assert "safety" in res["escalation_reason"].lower() or "misconduct" in res["escalation_reason"].lower()
    assert res["is_safety_escalation"] is True


def test_account_security_escalation():
    msg = "My account was deactivated suddenly and I cannot log in!"
    res = evaluate_escalation(
        customer_message=msg,
        intent="app_and_account_support",
        intent_confidence=0.90,
        evidence=[{"similarity": 0.80}],
    )
    assert res["decision"] == DECISION_ESCALATE
    assert "security" in res["escalation_reason"].lower()


def test_qualified_auto_handle():
    msg = "Where can I download my tax invoice receipt for my ride?"
    res = evaluate_escalation(
        customer_message=msg,
        intent="fare_and_billing_dispute",
        intent_confidence=0.88,
        evidence=[{"similarity": 0.78}],
    )
    assert res["decision"] == DECISION_AUTO_HANDLE
    assert res["escalation_reason"] is None
