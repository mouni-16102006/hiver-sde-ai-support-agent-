"""
Dataset loading and conversation extraction pipeline for Uber Support data.
Handles raw parquet/csv files, cached subsets, and reproducible offline fallbacks.
"""
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import (
    BRAND_HANDLE,
    DATA_DIR,
    EXTERNAL_PARQUET,
    HISTORICAL_KB_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    UBER_SUBSET_FILE,
)
from src.preprocessing import clean_tweet_text, extract_primary_exchange, is_english

logger = logging.getLogger(__name__)


def find_raw_dataset_path() -> Optional[Path]:
    """
    Searches standard local paths for raw Customer Support on Twitter datasets.
    """
    # 1. Processed Uber subset
    if UBER_SUBSET_FILE.exists():
        return UBER_SUBSET_FILE

    # 2. Local raw parquet in data/raw/
    raw_parquets = list(RAW_DATA_DIR.glob("*.parquet"))
    if raw_parquets:
        return raw_parquets[0]

    # 3. Known Hugging Face cache location
    if EXTERNAL_PARQUET.exists():
        return EXTERNAL_PARQUET

    # 4. Glob search across Hugging Face cache
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.exists():
        matches = list(hf_hub.glob("**/train-*.parquet"))
        if matches:
            return matches[0]

    # 5. Offline sample fallback
    sample_csv = RAW_DATA_DIR / "sample_uber_tweets.csv"
    if sample_csv.exists():
        return sample_csv

    return None


def extract_uber_pairs_from_raw(
    limit: Optional[int] = None,
    max_records: int = 15000
) -> pd.DataFrame:
    """
    Loads raw dataset, filters for Uber_Support, cleans, and extracts
    (customer_message, support_response) pairs.
    """
    raw_path = find_raw_dataset_path()
    if raw_path is None:
        raise FileNotFoundError(
            "No raw dataset found. Please ensure external parquet or sample_uber_tweets.csv exists."
        )

    logger.info(f"Loading data from: {raw_path}")
    if raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
        if "company" in df.columns:
            df = df[df["company"] == "Uber_Support"]
    elif raw_path.suffix == ".csv":
        df = pd.read_csv(raw_path)
        if "company" in df.columns:
            df = df[df["company"] == "Uber_Support"]
    else:
        raise ValueError(f"Unsupported file format: {raw_path}")

    records = []
    for idx, row in df.iterrows():
        raw_conv = row.get("conversation", "")
        if not isinstance(raw_conv, str):
            continue

        pair = extract_primary_exchange(raw_conv)
        if pair:
            records.append({
                "conversation_id": row.get("conversation_id", f"uber_{len(records)}"),
                "customer_message": pair["customer_message"],
                "support_response": pair["support_response"],
                "turn_count": pair["turn_count"],
            })

        if limit and len(records) >= limit:
            break
        if len(records) >= max_records:
            break

    df_pairs = pd.DataFrame(records)
    logger.info(f"Extracted {len(df_pairs)} high-quality customer-support pairs.")
    return df_pairs


def load_uber_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Loads processed Uber dataset if available; otherwise triggers extraction.
    """
    if UBER_SUBSET_FILE.exists():
        df = pd.read_parquet(UBER_SUBSET_FILE)
        if limit:
            return df.head(limit)
        return df

    return extract_uber_pairs_from_raw(limit=limit)


def load_historical_kb() -> List[Dict[str, str]]:
    """
    Loads historical knowledge base JSON list of resolution records.
    """
    if not HISTORICAL_KB_FILE.exists():
        return []
    with open(HISTORICAL_KB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
