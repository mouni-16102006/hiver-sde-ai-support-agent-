"""
Data preparation pipeline for Uber AI Support Agent.
Extracts, cleans, and structures Uber customer-support conversations from raw data.
Constructs historical knowledge base and sample fallback files.
"""
import json
import logging
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src.config import (
    DATA_DIR,
    EXTERNAL_PARQUET,
    HISTORICAL_KB_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    UBER_SUBSET_FILE,
)
from src.preprocessing import clean_tweet_text, extract_primary_exchange, is_english
from src.conversation_builder import validate_resolution_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Intent keyword rules for bootstrapping historical training tags
INTENT_TAG_RULES = {
    "cancellation_fee_issue": ["cancellation fee", "cancel fee", "charged for cancel", "cancelled", "canceling"],
    "lost_item_inquiry": ["lost", "left my", "forgot my", "phone in cab", "wallet in car", "left phone", "lost bag"],
    "driver_behavior_and_safety": ["safety", "reckless", "drunk", "accident", "rude", "harass", "abusive", "screamed", "refused"],
    "fare_and_billing_dispute": ["charge", "fare", "overcharged", "refund", "paytm", "money", "receipt", "toll", "bill", "double charge"],
    "pickup_and_route_issue": ["pickup", "location", "route", "gps", "arrived", "waiting", "wrong way", "driver didnt show"],
    "app_and_account_support": ["account", "login", "password", "promo", "code", "discount", "update phone", "deactivated", "blocked"],
}


def tag_intent(text: str) -> str:
    lower = text.lower()
    for intent, kws in INTENT_TAG_RULES.items():
        if any(kw in lower for kw in kws):
            return intent
    return "service_feedback_and_general"


def prepare_dataset(sample_limit: int = 15000):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    source_path = None
    if EXTERNAL_PARQUET.exists():
        source_path = EXTERNAL_PARQUET
        logger.info(f"Reading raw dataset from Hugging Face cache: {source_path}")
    else:
        # Check local raw dir
        local_files = list(RAW_DATA_DIR.glob("*.parquet")) + list(RAW_DATA_DIR.glob("*.csv"))
        if local_files:
            source_path = local_files[0]
            logger.info(f"Reading raw dataset from local file: {source_path}")

    if not source_path:
        raise FileNotFoundError("Could not find raw dataset in HF cache or data/raw/.")

    if source_path.suffix == ".parquet":
        df_raw = pd.read_parquet(source_path)
    else:
        df_raw = pd.read_csv(source_path)

    if "company" in df_raw.columns:
        df_uber = df_raw[df_raw["company"] == "Uber_Support"].copy()
    else:
        df_uber = df_raw

    logger.info(f"Loaded {len(df_uber)} total Uber_Support conversation threads.")

    extracted_records = []
    for idx, row in df_uber.iterrows():
        raw_conv = str(row.get("conversation", ""))
        pair = extract_primary_exchange(raw_conv)
        if not pair:
            continue

        c_msg = pair["customer_message"]
        s_msg = pair["support_response"]

        if not validate_resolution_pair(c_msg, s_msg):
            continue

        intent = tag_intent(c_msg)
        extracted_records.append({
            "conversation_id": row.get("conversation_id", f"uber_{len(extracted_records)}"),
            "customer_message": c_msg,
            "support_response": s_msg,
            "intent": intent,
            "turn_count": pair["turn_count"],
        })

        if len(extracted_records) >= sample_limit:
            break

    df_out = pd.DataFrame(extracted_records)
    logger.info(f"Successfully extracted and cleaned {len(df_out)} valid customer-support pairs.")

    # Save processed parquet
    df_out.to_parquet(UBER_SUBSET_FILE, index=False)
    logger.info(f"Saved processed parquet to {UBER_SUBSET_FILE}")

    # Save historical knowledge base JSON
    kb_records = df_out[["conversation_id", "customer_message", "support_response", "intent"]].to_dict(orient="records")
    with open(HISTORICAL_KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb_records, f, indent=2)
    logger.info(f"Saved historical KB with {len(kb_records)} records to {HISTORICAL_KB_FILE}")

    # Save offline sample fallback CSV in data/raw/
    sample_df = df_out.head(100)
    sample_csv = RAW_DATA_DIR / "sample_uber_tweets.csv"
    sample_df.to_csv(sample_csv, index=False)
    logger.info(f"Saved offline sample fallback CSV with 100 records to {sample_csv}")


if __name__ == "__main__":
    prepare_dataset()
