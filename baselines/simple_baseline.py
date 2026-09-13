"""
Simple Baseline for Uber AI Support Agent.
Implements a straightforward 1-Nearest Neighbor TF-IDF matching system
without calibrated confidence, safety keyword overrides, or grounded synthesis.
"""
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import BRAND_NAME, DECISION_AUTO_HANDLE, DECISION_ESCALATE
from src.preprocessing import clean_tweet_text


class SimpleBaselineAgent:
    """
    Simple non-LLM baseline: uncalibrated TF-IDF cosine retrieval that emits raw historical reply.
    """

    def __init__(self, documents: Optional[List[Dict]] = None):
        self.documents = documents or []
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        if self.documents:
            texts = [clean_tweet_text(d.get("customer_message", "")) for d in self.documents]
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def fit(self, documents: List[Dict]):
        self.documents = documents
        texts = [clean_tweet_text(d.get("customer_message", "")) for d in self.documents]
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def process_message(
        self,
        customer_message: str,
        exclude_texts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        cleaned = clean_tweet_text(customer_message)
        if self.matrix is None or not self.documents:
            return {
                "brand": BRAND_NAME,
                "customer_message": cleaned,
                "intent": "service_feedback_and_general",
                "intent_confidence": 0.40,
                "reply": "Please check help.uber.com.",
                "decision": DECISION_ESCALATE,
                "escalation_reason": "No index available",
                "evidence": [],
                "baseline_type": "simple_tfidf_1nn",
            }

        q_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        exclude_set = {clean_tweet_text(t).lower() for t in exclude_texts or []}
        best_idx = None
        best_sim = -1.0

        for i in sims.argsort()[::-1]:
            doc_text = clean_tweet_text(self.documents[i].get("customer_message", "")).lower()
            if doc_text in exclude_set:
                continue
            best_idx = i
            best_sim = float(sims[i])
            break

        if best_idx is not None and best_sim > 0.30:
            doc = self.documents[best_idx]
            reply = doc.get("support_response", "Please visit help.uber.com.")
            intent = doc.get("intent", "service_feedback_and_general")
            decision = DECISION_AUTO_HANDLE if best_sim >= 0.50 else DECISION_ESCALATE
            return {
                "brand": BRAND_NAME,
                "customer_message": cleaned,
                "intent": intent,
                "intent_confidence": round(best_sim, 4),
                "reply": reply,
                "decision": decision,
                "escalation_reason": None if decision == DECISION_AUTO_HANDLE else "Similarity below naive threshold",
                "evidence": [{
                    "customer_message": doc.get("customer_message", ""),
                    "support_response": doc.get("support_response", ""),
                    "similarity": round(best_sim, 4),
                    "intent": intent,
                }],
                "baseline_type": "simple_tfidf_1nn",
            }

        return {
            "brand": BRAND_NAME,
            "customer_message": cleaned,
            "intent": "service_feedback_and_general",
            "intent_confidence": 0.30,
            "reply": "Hi there, please visit help.uber.com for assistance.",
            "decision": DECISION_ESCALATE,
            "escalation_reason": "Low naive similarity",
            "evidence": [],
            "baseline_type": "simple_tfidf_1nn",
        }
