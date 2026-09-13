"""
Historical Support Evidence Retrieval module for Uber AI Support Agent.
Implements TF-IDF + Cosine Similarity search over verified historical customer-support pairs.
Includes explicit leakage prevention mechanisms.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    HISTORICAL_KB_FILE,
    RETRIEVER_INDEX_FILE,
    TOP_K_RETRIEVAL,
)
from src.preprocessing import clean_tweet_text

logger = logging.getLogger(__name__)


class HistoricalRetriever:
    """
    Retrieves top-k historically resolved customer-support interactions.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or RETRIEVER_INDEX_FILE
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.documents: List[Dict[str, str]] = []

        if self.index_path.exists():
            self.load()

    def build_index(self, kb_documents: List[Dict[str, str]]) -> "HistoricalRetriever":
        """
        Builds and vectorizes the historical knowledge base index.
        """
        self.documents = kb_documents
        texts = [clean_tweet_text(doc.get("customer_message", "")) for doc in kb_documents]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"Built retriever index over {len(self.documents)} historical documents.")
        return self

    def save(self, path: Optional[Path] = None):
        target = path or self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "documents": self.documents,
        }, target)
        logger.info(f"Saved retriever index to {target}")

    def load(self, path: Optional[Path] = None):
        target = path or self.index_path
        if not target.exists():
            raise FileNotFoundError(f"Retriever index not found at {target}")
        data = joblib.load(target)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.documents = data["documents"]
        logger.info(f"Loaded retriever index with {len(self.documents)} documents from {target}")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_intent: Optional[str] = None,
        exclude_texts: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieves top-k similar historical interactions.
        Supports filtering by intent and excluding queries to prevent evaluation leakage.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or not self.documents:
            return []

        cleaned_query = clean_tweet_text(query)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort indices in descending order
        ranked_indices = np.argsort(sims)[::-1]

        exclude_normalized = set()
        if exclude_texts:
            exclude_normalized = {clean_tweet_text(t).lower() for t in exclude_texts if t}

        results: List[Dict] = []
        for idx in ranked_indices:
            score = float(sims[idx])
            doc = self.documents[idx]
            cust_text = clean_tweet_text(doc.get("customer_message", "")).lower()

            # Prevent evaluation leakage
            if cust_text in exclude_normalized:
                continue

            # Optional intent filter
            if filter_intent and doc.get("intent") and doc.get("intent") != filter_intent:
                continue

            results.append({
                "customer_message": doc.get("customer_message", ""),
                "support_response": doc.get("support_response", ""),
                "intent": doc.get("intent", "unknown"),
                "similarity": round(score, 4),
            })

            if len(results) >= top_k:
                break

        return results
