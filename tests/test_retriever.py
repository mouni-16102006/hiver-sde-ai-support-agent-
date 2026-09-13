import pytest
from src.retriever import HistoricalRetriever


def test_retriever_query_and_exclusion():
    retriever = HistoricalRetriever()
    assert len(retriever.documents) > 0

    query = "How do I find a lost item left in the cab?"
    results = retriever.retrieve(query, top_k=3)
    assert len(results) > 0
    assert "customer_message" in results[0]
    assert "support_response" in results[0]
    assert 0.0 <= results[0]["similarity"] <= 1.0

    # Test leakage exclusion
    top_msg = results[0]["customer_message"]
    results_excluded = retriever.retrieve(query, top_k=3, exclude_texts=[top_msg])
    if results_excluded:
        assert results_excluded[0]["customer_message"] != top_msg
