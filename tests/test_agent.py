import pytest
from src.agent import UberAIAgent
from src.config import BRAND_NAME, DECISION_AUTO_HANDLE, DECISION_ESCALATE


def test_agent_end_to_end_structure():
    agent = UberAIAgent()
    query = "I was charged twice on my credit card for ride last night"
    res = agent.process_message(query)

    assert res["brand"] == BRAND_NAME
    assert res["intent"] in ["fare_and_billing_dispute", "cancellation_fee_issue"]
    assert 0.0 <= res["intent_confidence"] <= 1.0
    assert isinstance(res["reply"], str)
    assert len(res["reply"]) > 20
    assert res["decision"] in [DECISION_AUTO_HANDLE, DECISION_ESCALATE]
    assert isinstance(res["evidence"], list)
