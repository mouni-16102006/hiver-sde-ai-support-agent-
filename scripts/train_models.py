import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Model training and index construction script for Uber AI Support Agent.
Trains calibrated intent classifier and builds historical vector index
with strict exclusion of the Golden Evaluation Set to prevent data leakage.
"""
import json
import logging
import pandas as pd
from src.config import (
    CLASSIFIER_MODEL_FILE,
    GOLDEN_EVAL_FILE,
    HISTORICAL_KB_FILE,
    MODELS_DIR,
    RETRIEVER_INDEX_FILE,
    UBER_SUBSET_FILE,
)
from src.intent_classifier import UberIntentClassifier
from src.retriever import HistoricalRetriever
from src.preprocessing import clean_tweet_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_index():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Golden Set to construct strict exclusion set (Leakage Prevention)
    leakage_exclusion_set = set()
    if GOLDEN_EVAL_FILE.exists():
        with open(GOLDEN_EVAL_FILE, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
            for item in gold_data:
                leakage_exclusion_set.add(clean_tweet_text(item["customer_message"]).lower())
        logger.info(f"Loaded {len(leakage_exclusion_set)} golden examples to exclude from training and index.")

    # 2. Load Processed Uber Data for Training
    if not UBER_SUBSET_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found at {UBER_SUBSET_FILE}. Run prepare_data.py first.")

    df = pd.read_parquet(UBER_SUBSET_FILE)
    logger.info(f"Total available candidate pairs: {len(df)}")

    # Filter out golden evaluation set items
    train_records = []
    for idx, row in df.iterrows():
        c_msg = clean_tweet_text(row["customer_message"])
        if c_msg.lower() not in leakage_exclusion_set:
            train_records.append({
                "conversation_id": row["conversation_id"],
                "customer_message": c_msg,
                "support_response": row["support_response"],
                "intent": row["intent"],
            })

    logger.info(f"Clean training & retrieval pool after leakage exclusion: {len(train_records)} records.")

    # 3. Train Intent Classifier
    train_texts = [r["customer_message"] for r in train_records]
    train_labels = [r["intent"] for r in train_records]

    classifier = UberIntentClassifier()
    classifier.train(train_texts, train_labels)
    classifier.save(CLASSIFIER_MODEL_FILE)
    logger.info(f"Saved trained intent classifier to {CLASSIFIER_MODEL_FILE}")

    # 4. Build Historical Retriever Index
    retriever = HistoricalRetriever()
    retriever.build_index(train_records)
    retriever.save(RETRIEVER_INDEX_FILE)
    logger.info(f"Saved retriever index to {RETRIEVER_INDEX_FILE}")


if __name__ == "__main__":
    train_and_index()
