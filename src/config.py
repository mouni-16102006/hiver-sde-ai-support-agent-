"""
System Configuration for Uber AI Support Agent.
Defines directories, intent taxonomies, operational thresholds, and brand metadata.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_DATA_DIR = DATA_DIR / "golden"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Brand Identification
BRAND_NAME = "Uber"
BRAND_HANDLE = "@Uber_Support"
PROJECT_NAME = "Uber AI Support Agent"
DISCLAIMER = "Independent prototype — not an official Uber support system."

# File Paths
UBER_SUBSET_FILE = PROCESSED_DATA_DIR / "uber_conversations.parquet"
HISTORICAL_KB_FILE = PROCESSED_DATA_DIR / "historical_kb.json"
GOLDEN_EVAL_FILE = GOLDEN_DATA_DIR / "golden_eval_set.json"
GOLDEN_EVAL_CSV = GOLDEN_DATA_DIR / "golden_eval_set.csv"
HUMAN_TEMPLATE_CSV = GOLDEN_DATA_DIR / "human_labels_template.csv"
CLASSIFIER_MODEL_FILE = MODELS_DIR / "intent_classifier.joblib"
RETRIEVER_INDEX_FILE = MODELS_DIR / "retriever_index.joblib"
EVAL_RESULTS_JSON = REPORTS_DIR / "evaluation_results.json"

# External Dataset Cache
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
EXTERNAL_PARQUET = HF_CACHE_DIR / "datasets--TNE-AI--customer-support-on-twitter-conversation" / "snapshots" / "878553003db208b7cb8797b26a43a461896db940" / "data" / "train-00000-of-00001.parquet"

# Intent Taxonomy (7 Core Intents derived from Uber Support Data)
INTENTS = [
    "fare_and_billing_dispute",
    "cancellation_fee_issue",
    "driver_behavior_and_safety",
    "lost_item_inquiry",
    "pickup_and_route_issue",
    "app_and_account_support",
    "service_feedback_and_general",
]

INTENT_DESCRIPTIONS = {
    "fare_and_billing_dispute": "Disputes over charged amounts, toll fees, payment failures, duplicate deductions, or incorrect fare calculation.",
    "cancellation_fee_issue": "Inquiries or fee waiver requests regarding cancellation charges applied when driver or rider cancelled.",
    "driver_behavior_and_safety": "Reports of unprofessional behavior, reckless driving, route refusal, verbal altercations, or physical safety concerns.",
    "lost_item_inquiry": "Queries regarding lost personal belongings (phone, wallet, keys, bag) left in a vehicle and driver contact.",
    "pickup_and_route_issue": "Issues with driver arrival, inaccurate GPS pin, long waiting times, or driver taking an excessively long circuitous route.",
    "app_and_account_support": "Account access, login errors, phone number verification, deactivation appeals, or promo code application issues.",
    "service_feedback_and_general": "General praise, feedback, operating city queries, or general service inquiries.",
}

# Operational Decision Enums
DECISION_AUTO_HANDLE = "AUTO_HANDLE"
DECISION_ESCALATE = "ESCALATE_TO_HUMAN"

# Operational Calibration Thresholds
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.70"))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("ESCALATION_SIMILARITY_THRESHOLD", "0.60"))
TOP_K_RETRIEVAL = int(os.getenv("MAX_RETRIEVAL_RESULTS", "3"))

# Critical Safety and Security Triggers (Always Escalate)
CRITICAL_SAFETY_KEYWORDS = [
    "safety", "reckless", "drunk", "accident", "harass", "assault",
    "police", "emergency", "threat", "weapon", "injury", "injuries",
    "sexual", "physical", "attack", "hit and run", "screamed", "danger"
]

ACCOUNT_SECURITY_KEYWORDS = [
    "deactivated", "blocked account", "account hacked", "unauthorized transaction",
    "fraud", "stolen credit card", "banned"
]
