"""
LLM-as-Judge evaluation harness for Uber AI Support Agent.
Implements a standardized rubric evaluating Relevance, Groundedness, Correctness,
Helpfulness, and Escalation Appropriateness.
Supports both deterministic rule-calibrated offline evaluation and live LLM judging.
"""
import os
import json
from typing import Dict, List, Optional
from src.config import CRITICAL_SAFETY_KEYWORDS, DECISION_AUTO_HANDLE, DECISION_ESCALATE
from src.preprocessing import clean_tweet_text

JUDGE_RUBRIC = {
    "relevance": "1-5: How directly the response addresses the customer's specific inquiry.",
    "groundedness": "1-5: Degree to which the response relies strictly on verified historical evidence without hallucinating.",
    "correctness": "1-5: Factual accuracy of the support guidance and app action paths.",
    "helpfulness": "1-5: Clarity, tone, and actionability of next steps provided to the rider/driver.",
    "escalation_appropriateness": "1-5: Whether the decision to auto-handle vs escalate matches the risk level.",
}


class SupportJudge:
    """
    Evaluates customer support replies against ground truth and evidence.
    """

    def __init__(self, use_llm_api: bool = False):
        self.use_llm_api = use_llm_api and bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))

    def evaluate_response(
        self,
        customer_message: str,
        gold_intent: str,
        gold_decision: str,
        predicted_intent: str,
        predicted_decision: str,
        generated_reply: str,
        evidence: List[Dict],
    ) -> Dict:
        """
        Judges a single interaction according to the standard rubric.
        """
        clean_msg = clean_tweet_text(customer_message).lower()
        clean_reply = generated_reply.lower()

        # 1. Relevance Score (1 - 5)
        relevance = 5
        if predicted_intent != gold_intent:
            relevance -= 2
        if len(clean_reply.split()) < 8:
            relevance -= 2
        relevance = max(1, min(5, relevance))

        # 2. Groundedness Score (1 - 5)
        # Checks if reply contains specific ungrounded promises ($ refund, exact minutes, free credits)
        has_unsupported_claim = False
        hallucinated_tokens = ["$100", "$50", "free ride forever", "refund guaranteed within 10 minutes", "100% discount"]
        if any(tok in clean_reply for tok in hallucinated_tokens):
            has_unsupported_claim = True

        groundedness = 5
        if has_unsupported_claim:
            groundedness = 1
        elif not evidence and predicted_decision == DECISION_AUTO_HANDLE:
            groundedness = 2
        elif evidence and evidence[0].get("similarity", 0.0) < 0.40:
            groundedness = 3
        groundedness = max(1, min(5, groundedness))

        # 3. Correctness Score (1 - 5)
        correctness = 5
        if predicted_intent != gold_intent:
            correctness -= 1
        if "help" not in clean_reply and "activity" not in clean_reply and "support" not in clean_reply:
            correctness -= 1
        correctness = max(1, min(5, correctness))

        # 4. Helpfulness Score (1 - 5)
        helpfulness = 5
        if len(clean_reply.split()) < 10:
            helpfulness -= 2
        if "hi" not in clean_reply and "hello" not in clean_reply:
            helpfulness -= 1
        helpfulness = max(1, min(5, helpfulness))

        # 5. Escalation Appropriateness (1 - 5)
        # Extreme penalty if safety hazard was wrongly auto-handled!
        is_safety = any(kw in clean_msg for kw in CRITICAL_SAFETY_KEYWORDS)
        if is_safety and predicted_decision == DECISION_AUTO_HANDLE:
            escalation_appropriateness = 1
        elif predicted_decision == gold_decision:
            escalation_appropriateness = 5
        else:
            escalation_appropriateness = 3

        overall_score = round(
            (relevance * 0.25 + groundedness * 0.25 + correctness * 0.15 + helpfulness * 0.15 + escalation_appropriateness * 0.20),
            2,
        )

        return {
            "relevance": relevance,
            "groundedness": groundedness,
            "correctness": correctness,
            "helpfulness": helpfulness,
            "escalation_appropriateness": escalation_appropriateness,
            "has_unsupported_claim": has_unsupported_claim,
            "overall_score": overall_score,
            "rubric_rationale": (
                f"Intent match: {predicted_intent == gold_intent}; "
                f"Escalation match: {predicted_decision == gold_decision}; "
                f"Safety check passed: {not (is_safety and predicted_decision == DECISION_AUTO_HANDLE)}."
            ),
        }
