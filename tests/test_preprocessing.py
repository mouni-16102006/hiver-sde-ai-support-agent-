import pytest
from src.preprocessing import clean_tweet_text, is_english, parse_turns, extract_primary_exchange


def test_clean_tweet_text():
    raw = "@Uber_Support I was charged twice for my ride! https://t.co/xyz123"
    cleaned = clean_tweet_text(raw)
    assert "@Uber_Support" not in cleaned
    assert "[URL]" in cleaned
    assert "charged twice" in cleaned


def test_is_english():
    assert is_english("Why was my credit card debited twice?") is True
    assert is_english("مرحبا بك في اوبر") is False


def test_parse_turns():
    conv = "Customer: Where is my driver?\nSupport: Driver is arriving in 2 mins."
    turns = parse_turns(conv)
    assert len(turns) == 2
    assert turns[0] == ("Customer", "Where is my driver?")
    assert turns[1] == ("Support", "Driver is arriving in 2 mins.")


def test_extract_primary_exchange():
    conv = "Customer: I left my iPhone in the cab.\nSupport: Contact the driver via the app Activity tab."
    pair = extract_primary_exchange(conv)
    assert pair is not None
    assert "iPhone" in pair["customer_message"]
    assert "Activity tab" in pair["support_response"]
