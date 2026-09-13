import pytest
from evaluation.metrics import compute_intent_metrics, compute_escalation_metrics
from src.config import DECISION_AUTO_HANDLE, DECISION_ESCALATE


def test_compute_intent_metrics():
    y_true = ["fare_and_billing_dispute", "lost_item_inquiry", "cancellation_fee_issue"]
    y_pred = ["fare_and_billing_dispute", "lost_item_inquiry", "fare_and_billing_dispute"]
    m = compute_intent_metrics(y_true, y_pred)
    assert m["accuracy"] == round(2 / 3, 4)
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_compute_escalation_metrics_safety():
    # 2 cases that should escalate, 1 auto-handled
    y_true = [DECISION_ESCALATE, DECISION_ESCALATE, DECISION_AUTO_HANDLE]
    y_pred = [DECISION_ESCALATE, DECISION_AUTO_HANDLE, DECISION_AUTO_HANDLE]

    m = compute_escalation_metrics(y_true, y_pred)
    # 1 of the 2 escalations was wrongly auto-handled -> false auto handle rate is 50%
    assert m["false_auto_handle_rate"] == 0.50
    assert m["false_auto_handles_count"] == 1
