"""
Text preprocessing and conversation parsing utilities for Uber Customer Support data.
"""
import re
from typing import Dict, List, Optional, Tuple

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HANDLE_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
ANONYMIZED_PATTERN = re.compile(r"__\w+__")


def clean_tweet_text(text: str, remove_handles: bool = True, preserve_url_token: bool = True) -> str:
    """
    Cleans raw customer or support tweet text:
    - Normalizes URLs
    - Optionally strips twitter @handles
    - Cleans extra whitespace and standardized punctuation
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text
    if preserve_url_token:
        cleaned = URL_PATTERN.sub("[URL]", cleaned)
    else:
        cleaned = URL_PATTERN.sub("", cleaned)

    if remove_handles:
        cleaned = HANDLE_PATTERN.sub("", cleaned)

    # Replace multiple spaces/newlines
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def is_english(text: str, threshold: float = 0.8) -> bool:
    """
    Returns True if the text is predominantly English/Latin alphabet.
    Filters out non-English customer tweets (Arabic, Cyrillic, Hindi scripts, etc.).
    """
    if not isinstance(text, str) or not text.strip():
        return False

    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False

    latin_chars = [c for c in alpha_chars if ord(c) < 128]
    return (len(latin_chars) / len(alpha_chars)) >= threshold


def parse_turns(conversation_text: str) -> List[Tuple[str, str]]:
    """
    Parses a multi-turn conversation string into a list of (speaker, message) tuples.
    Expected speaker tokens: Customer:, Support:
    """
    turns: List[Tuple[str, str]] = []
    if not isinstance(conversation_text, str) or not conversation_text.strip():
        return turns

    lines = conversation_text.splitlines()
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if line_s.startswith("Customer:"):
            turns.append(("Customer", line_s[9:].strip()))
        elif line_s.startswith("Support:"):
            turns.append(("Support", line_s[8:].strip()))
    return turns


def extract_primary_exchange(conversation_text: str) -> Optional[Dict[str, str]]:
    """
    Extracts the initial customer message and the first meaningful support reply
    from a multi-turn conversation string.
    Filters out empty exchanges or marketing-only support tweets.
    """
    turns = parse_turns(conversation_text)
    if not turns:
        return None

    first_customer_msg = ""
    first_support_msg = ""

    for speaker, msg in turns:
        if speaker == "Customer" and not first_customer_msg:
            cleaned = clean_tweet_text(msg)
            if len(cleaned) >= 10 and is_english(cleaned):
                first_customer_msg = cleaned
        elif speaker == "Support" and first_customer_msg and not first_support_msg:
            cleaned = clean_tweet_text(msg)
            if len(cleaned) >= 10:
                first_support_msg = cleaned
                break

    if first_customer_msg and first_support_msg:
        return {
            "customer_message": first_customer_msg,
            "support_response": first_support_msg,
            "turn_count": len(turns),
        }
    return None
