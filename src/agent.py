"""
Main AI Support Agent orchestration module for Uber Support.
Coordinates preprocessing, intent classification, evidence retrieval,
escalation decisioning, and grounded response composition.
"""
from typing import Dict, List, Optional
from src.config import BRAND_NAME, PROJECT_NAME
from src.intent_classifier import UberIntentClassifier
from src.retriever import HistoricalRetriever
from src.escalation import evaluate_escalation
from src.reply_generator import GroundedReplyGenerator
from src.preprocessing import clean_tweet_text


class UberAIAgent:
    """
    End-to-end production pipeline for Uber AI Support Agent.
    """

    def __init__(
        self,
        classifier: Optional[UberIntentClassifier] = None,
        retriever: Optional[HistoricalRetriever] = None,
        generator: Optional[GroundedReplyGenerator] = None,
    ):
        self.classifier = classifier or UberIntentClassifier()
        self.retriever = retriever or HistoricalRetriever()
        self.generator = generator or GroundedReplyGenerator()

    def process_message(
        self,
        customer_message: str,
        exclude_texts: Optional[List[str]] = None,
        filter_intent: bool = False,
    ) -> Dict:
        """
        Executes full agent pipeline for a customer inquiry.
        """
        cleaned_query = clean_tweet_text(customer_message)

        # 1. Intent Classification
        intent, confidence = self.classifier.predict(cleaned_query)

        # 2. Historical Evidence Retrieval
        intent_filter = intent if filter_intent else None
        evidence = self.retriever.retrieve(
            query=cleaned_query,
            filter_intent=intent_filter,
            exclude_texts=exclude_texts,
        )

        # 3. Escalation Decision & Auditing
        escalation_info = evaluate_escalation(
            customer_message=cleaned_query,
            intent=intent,
            intent_confidence=confidence,
            evidence=evidence,
        )

        decision = escalation_info["decision"]
        escalation_reason = escalation_info["escalation_reason"]

        # 4. Grounded Response Generation
        reply = self.generator.generate_reply(
            customer_message=cleaned_query,
            intent=intent,
            evidence=evidence,
            decision=decision,
            escalation_reason=escalation_reason,
        )

        # 5. Return Structured JSON Agent Output
        return {
            "brand": BRAND_NAME,
            "customer_message": cleaned_query,
            "intent": intent,
            "intent_confidence": confidence,
            "reply": reply,
            "decision": decision,
            "escalation_reason": escalation_reason,
            "evidence": evidence,
            "rule_triggered": escalation_info.get("rule_triggered"),
            "is_safety_escalation": escalation_info.get("is_safety_escalation", False),
        }
