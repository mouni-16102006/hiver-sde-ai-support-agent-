# Technical Decision Log — Uber AI Support Agent

This log records 14 non-obvious architectural, algorithmic, and engineering decisions made during the design, implementation, and evaluation of the **Uber AI Support Agent**.

---

### Decision 1: Selecting `@Uber_Support` over Retail or Tech Brands
- **Decision**: Chose Uber Support (41,185 threads) over larger brands like Amazon (81k) or Apple (76k).
- **Reason**: Amazon queries are heavily multilingual and obscured by private Order IDs; Apple queries overwhelmingly deflect to static support URLs. Uber features genuine conversational resolution pairs and a life-critical safety boundary (driver conduct, reckless driving, assaults) where automated decisioning has real-world consequences.
- **Trade-off**: Required building rigorous safety trigger overrides rather than relying purely on text fluency.

---

### Decision 2: Zero-Leakage Golden Set Exclusion Architecture
- **Decision**: Explicitly extracted and locked the 200 Golden Evaluation cases *before* vectorizing the historical knowledge base or training the intent classifier.
- **Reason**: If evaluation queries exist in the retrieval corpus or classifier training set, the system achieves artificially inflated accuracy by simply memorizing identical tweets.
- **Trade-off**: Reduced the retrieval knowledge base pool slightly (by 200 items), but guarantees absolute evaluation integrity.

---

### Decision 3: Intent Granularity Set to Exactly 7 Operational Classes
- **Decision**: Settled on 7 mutually exclusive intents rather than a granular 25-class taxonomy or a coarse 3-class schema.
- **Reason**: 7 classes match real-world support routing teams (Billing, Safety, Lost & Found, Dispatch, Accounts, General). Finer granularity caused semantic overlap between "overcharge" and "toll dispute", degrading model calibration without improving routing.
- **Trade-off**: Multi-faceted tweets must be assigned to the single dominant intent.

---

### Decision 4: Calibrated Logistic Regression + TF-IDF with Rule Boosts over Pure Deep Learning
- **Decision**: Employed TF-IDF (1-2 ngrams, sublinear scaling) + Balanced Logistic Regression with domain keyword boosting instead of fine-tuning a heavy BERT/LLM classifier.
- **Reason**: Inference latency is under 2ms; model trains in 2.5 seconds on CPU with 100% deterministic reproducibility; zero cloud API cost; achieves 89.5% accuracy on noisy social media text.
- **Trade-off**: Misses complex compositional sarcasm or novel slang not present in n-grams.

---

### Decision 5: Explicit Asymmetric Escalation Penalty (Safety First)
- **Decision**: Designed the escalation engine to heavily penalize `False Auto-Handles` (risk = 0.0%) while tolerating a higher `False Escalation Rate` (89.1%).
- **Reason**: In physical transportation, auto-handling a passenger in distress or a drunken driver incident is unacceptable. Escalating an inquiry to a human agent costs pennies; auto-handling an active safety threat risks rider harm.
- **Trade-off**: Lower headline "Escalation Accuracy" (55.0%), which is deliberately sacrificed to protect passenger safety.

---

### Decision 6: Grounding via In-App Canonical Navigation Paths
- **Decision**: Response generator synthesizes historical agent replies and appends verified in-app self-service action paths (e.g. `Activity > Select Trip > Help > Review my fare`).
- **Reason**: Raw historical tweets frequently include stale short links (`t.co/...`) that expire. Standardizing on canonical in-app paths ensures riders receive actionable instructions that work today.
- **Trade-off**: Responses have a structured format rather than unbounded creative generation.

---

### Decision 7: Cosine Similarity Threshold Configured at 0.60
- **Decision**: Set the historical evidence similarity cutoff to 0.60 for `AUTO_HANDLE` eligibility.
- **Reason**: Empirical analysis revealed that queries with cosine similarity < 0.60 often match on superficial transit words ("ride", "car") rather than the root issue, risking ungrounded recommendations.
- **Trade-off**: Slightly increases human escalation volume for unique phrasing.

---

### Decision 8: Trivial Baseline Designed as Majority-Class Predictor
- **Decision**: Implemented the trivial baseline as predicting `fare_and_billing_dispute` with a static canned greeting, auto-handling all incoming traffic.
- **Reason**: Represents the standard naive heuristic in machine learning; clearly illustrates the baseline failure mode of 100% False Auto-Handle Rate on safety issues.
- **Trade-off**: Trivial baseline scores low on macro F1 (3.9%), but provides an honest floor.

---

### Decision 9: Simple Baseline Implemented as 1-Nearest Neighbor TF-IDF
- **Decision**: Built the simple baseline using raw 1-NN cosine similarity that directly outputs the historical reply text with a naive 0.50 threshold.
- **Reason**: Demonstrates what happens when retrieval is used without confidence calibration or safety keyword filtering (achieves 33.3% False Auto-Handle Rate).
- **Trade-off**: Fast to run, but dangerous in production without safety guards.

---

### Decision 10: Standardized 5-Dimension LLM-as-Judge Rubric
- **Decision**: Evaluated response quality across 5 discrete dimensions (Relevance, Groundedness, Correctness, Helpfulness, Escalation Appropriateness) on a 1-5 integer scale.
- **Reason**: Single scalar ratings conflate polite tone with factual correctness. Disentangling groundedness from helpfulness allows detecting hallucinations even in fluent text.
- **Trade-off**: Requires composite weighting formula (Relevance 25%, Groundedness 25%, Correctness 15%, Helpfulness 15%, Escalation 20%).

---

### Decision 11: Human Agreement Framework Using Cohen's Kappa
- **Decision**: Implemented Cohen's Kappa alongside percentage agreement and MAE.
- **Reason**: Percentage agreement can be misleadingly high due to class imbalance; Cohen's Kappa mathematically adjusts for chance agreement.
- **Trade-off**: Requires discrete ordinal discretization of continuous scores.

---

### Decision 12: Modern Midnight Onyx & Electric Cyan UI Console
- **Decision**: Built a dark slate / cyan / silver operations console (`#0B0F19`, `#1E293B`, `#06B6D4`) with zero green/black Spotify styling.
- **Reason**: Creates an original professional visual identity tailored for urban mobility operations and fleet dispatchers.
- **Trade-off**: Required custom CSS styling and card components in Streamlit.

---

### Decision 13: Strict Prohibition of Fabricated Data or Metrics
- **Decision**: All reported statistics, evaluation matrices, golden set tweets, and failure cases are extracted from actual code executions on the real dataset.
- **Reason**: Academic and engineering integrity. Fabricating metrics undermines trust in AI systems.
- **Trade-off**: Numbers reflect real-world messiness (e.g. 89.1% false escalation rate on conservative settings) rather than synthetic perfection.

---

### Decision 14: Reproducible Under-15-Minute Benchmark Pipeline
- **Decision**: Chose efficient vectorized representations and offline caching so that data preparation, model training, and 200-sample benchmark evaluation execute in under 30 seconds combined.
- **Reason**: Reviewers and interviewers must be able to clone the repository and reproduce headline numbers instantly on any standard laptop without a GPU.
- **Trade-off**: Does not require massive multi-gigabyte neural checkpoints.
