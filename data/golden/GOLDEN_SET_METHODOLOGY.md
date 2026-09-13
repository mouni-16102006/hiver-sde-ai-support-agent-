# Golden Evaluation Set Methodology

## 1. Overview & Purpose
This golden evaluation set contains **200 hand-curated and labelled customer-support interactions** exclusively from Uber Support (@Uber_Support). It serves as the authoritative, leak-free benchmark for evaluating intent classification, evidence retrieval, and safety-critical escalation decisions.

## 2. Sampling Strategy
- **Source Data**: Kaggle / Hugging Face Customer Support on Twitter dataset filtered strictly to `@Uber_Support` (41,185 threads).
- **Stratified Intent Coverage**: Sampled across all 7 operational intents to ensure representation of both high-volume queries (fare disputes, cancellation fees) and rare but safety-critical queries (reckless driving, lost passports).
- **Leakage Prevention**: All golden evaluation conversations are explicitly excluded from the training dataset and the retrieval knowledge base index.

## 3. Intent Distribution in Golden Set
| Intent | Count |
|---|---|
| fare_and_billing_dispute | 32 |
| cancellation_fee_issue | 30 |
| driver_behavior_and_safety | 30 |
| lost_item_inquiry | 30 |
| pickup_and_route_issue | 29 |
| app_and_account_support | 29 |
| service_feedback_and_general | 20 |

## 4. Escalation Distribution
- **AUTO_HANDLE**: 101 (50.5%)
- **ESCALATE_TO_HUMAN**: 99 (49.5%)

## 5. Labelling Guidelines & Safety Rubric
- **Safety Hazards & Misconduct**: Any report of reckless driving, intoxication, verbal abuse, sexual harassment, or physical altercations is tagged as `ESCALATE_TO_HUMAN`. Auto-handling a safety hazard is treated as a critical system failure (`False Auto-Handle`).
- **Account Security & Fraud**: Unauthorized transactions, account takeovers, or deactivations require human verification.
- **Routine Self-Service**: Standard cancellation fee waivers, lost item driver contact steps, and receipt download requests are tagged as `AUTO_HANDLE`.
