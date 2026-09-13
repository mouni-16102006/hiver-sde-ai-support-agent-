"""
Conversation builder module. Reconstructs multi-turn dialogue structures,
validates quality, and pairs customer inquiries with canonical support resolutions.
"""
from typing import Dict, List, Optional
from src.preprocessing import clean_tweet_text, parse_turns, is_english


def build_conversation_thread(raw_text: str) -> Dict:
    """
    Parses full conversation into structured turns with metadata.
    """
    turns = parse_turns(raw_text)
    structured_turns = []
    for speaker, text in turns:
        cleaned = clean_tweet_text(text)
        if cleaned:
            structured_turns.append({
                "speaker": speaker,
                "text": cleaned,
                "is_english": is_english(cleaned),
                "length": len(cleaned),
            })

    return {
        "turn_count": len(structured_turns),
        "turns": structured_turns,
        "is_valid": len(structured_turns) >= 2,
    }


def validate_resolution_pair(customer_msg: str, support_msg: str) -> bool:
    """
    Validates that customer inquiry and support reply form a legitimate, non-trivial interaction.
    """
    if not customer_msg or not support_msg:
        return False
    if len(customer_msg.split()) < 3:
        return False
    if len(support_msg.split()) < 4:
        return False
    # Avoid generic bot loop
    if customer_msg.strip().lower() == support_msg.strip().lower():
        return False
    return True
