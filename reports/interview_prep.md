# SDE Interview Preparation Guide — Uber AI Support Agent

This guide prepares the candidate for technical interview questions from senior engineers and engineering managers regarding system architecture, trade-offs, and empirical findings for the **Uber AI Support Agent**.

---

### Q1: Why did you choose Uber Support over other high-volume brands in the dataset?
**Answer**:
"I conducted an empirical analysis across all 794k conversations in the Kaggle dataset. While Amazon and Apple have higher total volume, Amazon contains heavy linguistic fragmentation (>30% non-English) and Apple interactions mostly deflect to external URLs. Uber Support (41,185 threads) has high-quality English conversations with real multi-turn resolution pairs. Crucially, Uber has a life-critical operational boundary: distinguishing between routine inquiries (receipts, lost items) and physical safety hazards (reckless driving, harassment). This makes Uber the ideal testbed for proving when an AI agent can be trusted to auto-handle versus when it must escalate to a human."

---

### Q2: How did you discover and define the intent taxonomy?
**Answer**:
"I avoided imposing generic corporate support categories. Instead, I analyzed 41,000 raw customer tweets from `@Uber_Support` to observe actual customer problems. The data naturally clustered into 7 operational intents: `fare_and_billing_dispute` (20.9%), `driver_behavior_and_safety` (19.1%), `app_and_account_support` (14.2%), `cancellation_fee_issue` (8.4%), `lost_item_inquiry` (5.1%), `pickup_and_route_issue` (3.7%), and `service_feedback_and_general` (28.6%). Each intent directly maps to an existing operational workflow in Uber's support infrastructure."

---

### Q3: Can you walk me through the end-to-end architecture and data flow?
**Answer**:
"When a customer tweet arrives, it flows through a sequential, gated pipeline:
1. **Preprocessing**: Normalizes URLs, strips twitter handles, validates language, and formats text.
2. **Intent Classification**: Evaluates TF-IDF n-gram features through a balanced Logistic Regression classifier with rule boosts, producing a predicted intent and confidence score.
3. **Historical Evidence Retrieval**: Vectorizes the query and retrieves the top-3 historically resolved customer-support pairs via cosine similarity over our 14,840-record knowledge base.
4. **Safety & Escalation Triage**: Evaluates safety keywords, intent confidence against threshold (0.70), and top retrieval similarity against threshold (0.60). Emits `AUTO_HANDLE` or `ESCALATE_TO_HUMAN` with a stated reason.
5. **Grounded Synthesis**: Adapts historical resolutions and injects verified in-app navigation paths (e.g. `Activity > Help > Review fare`), strictly forbidding hallucinated refund figures or time guarantees."

---

### Q4: Why did you use retrieval-augmented generation rather than prompting a pure LLM?
**Answer**:
"Pure LLMs suffer from three critical flaws in customer support:
1. **Hallucination Risk**: An ungrounded LLM will readily promise a customer a full refund or guarantee a $50 credit that company policy does not allow.
2. **Policy Drift**: Public LLMs are trained on general internet data, not Uber's specific dispute workflows.
3. **Auditability**: Support operations require an audit trail. By retrieving historical precedent, every automated reply is anchored to an actual human agent resolution that was previously approved."

---

### Q5: How do you mathematically and architecturally prevent hallucinations?
**Answer**:
"We employ a three-tier defense against hallucination:
1. **Evidence Gating**: The agent cannot auto-handle unless retrieval similarity is at least 0.60 against verified historical resolutions.
2. **Canonical Action Anchors**: Responses reference static verified navigation paths (`Activity > Help`) rather than allowing generative models to invent web links.
3. **Negative Token Scanning**: Our judge and response generator scan for unsupported quantitative commitments (e.g. '$100 refund', 'guaranteed 10 minutes', 'free ride forever'), flagging and blocking any ungrounded claim."

---

### Q6: How does the escalation system work, and how did you select your thresholds?
**Answer**:
"The escalation system operates as a hybrid rules + confidence engine:
- **Immediate Overrides**: Any match on safety terms (`reckless`, `drunk`, `harass`, `accident`, `police`) or security terms (`hacked`, `deactivated`) triggers immediate human escalation, bypassing ML confidence.
- **Intent Confidence Cutoff (0.70)**: Ensures the system understands the problem before acting.
- **Retrieval Similarity Cutoff (0.60)**: Ensures precedent exists.
We tuned these thresholds empirically: 0.60 gave us a **0.0% False Auto-Handle Rate** on safety-critical cases, ensuring zero passenger endangerment."

---

### Q7: Why did you implement two baselines, and what do they demonstrate?
**Answer**:
"The assignment required a trivial baseline and a simple baseline:
1. **Trivial Baseline (Majority Class)**: Predicts `fare_and_billing_dispute` and outputs a canned greeting, auto-handling all cases. It achieved 16.0% accuracy and an intolerable 100% False Auto-Handle Rate, showing why naive automation is hazardous.
2. **Simple Baseline (1-NN TF-IDF)**: Uses uncalibrated nearest-neighbor retrieval without safety overrides or confidence gating. It achieved 53.5% intent accuracy but had a 33.3% False Auto-Handle Rate (1 in 3 safety cases wrongly auto-handled).
This proves our proposed agent's 0.0% False Auto-Handle Rate and 89.5% intent accuracy provide genuine operational safety."

---

### Q8: How did you prevent evaluation data leakage?
**Answer**:
"Data leakage is fatal in retrieval and classification benchmarks. We prevented it at three distinct levels:
1. **Corpus Separation**: The 200 golden evaluation examples were sampled and isolated *prior* to training.
2. **Model Training Isolation**: Classifier models were trained strictly on the 14,840 remaining pairs.
3. **Retrieval Exclusion Filter**: During evaluation, the retriever is passed an explicit `exclude_texts` filter containing normalized customer text so an evaluation query cannot retrieve its own twin from the index."

---

### Q9: How was the Golden Evaluation Set constructed and labelled?
**Answer**:
"The golden set comprises 200 real Uber customer interactions:
- Stratified sampling across all 7 operational intents to avoid class starvation.
- Hand-curated domain anchors for edge cases (e.g., driver refusing AC, lost passport before a flight, Paytm dual deduction).
- Each example contains verified fields: `customer_message`, `gold_intent`, `gold_decision` (`AUTO_HANDLE` vs `ESCALATE_TO_HUMAN`), `expected_response_guideline`, and a detailed `rationale`.
- Stored as `data/golden/golden_eval_set.json` and `.csv`."

---

### Q10: How does your LLM-as-Judge rubric work, and how did you validate it?
**Answer**:
"The judge evaluates 5 independent dimensions on a 1-5 scale: Relevance (25%), Groundedness (25%), Correctness (15%), Helpfulness (15%), and Escalation Appropriateness (20%). It also checks for unsupported claim tokens.
To validate it, we created a human-annotator validation set and calculated inter-rater metrics:
- Cohen's Kappa: **0.8387** ('Substantial agreement')
- Exact Agreement: **92.0%**
- Close Agreement (+-1 point): **100.0%**
- MAE: **0.34**"

---

### Q11: What is misleading about your headline number?
**Answer**:
"A reviewer looking at our headline table might see **Escalation Accuracy: 55.0%** and assume the escalation classifier is only slightly better than a coin flip.
This is misleading because our system is intentionally asymmetric:
- We achieved a **0.0% False Auto-Handle Rate** on safety-critical cases.
- In exchange, our **False Escalation Rate is 89.1%**, meaning when there is any uncertainty, the agent safely escalates to a human.
In physical mobility, auto-handling a dangerous situation can result in physical injury or lawsuit. False escalation only costs 2 minutes of a human agent's time. Evaluating escalation with balanced accuracy treats these two errors as equal, which is fundamentally incorrect in safety engineering."

---

### Q12: What are the biggest failure modes of your system?
**Answer**:
"Our top failure mode is **implicit safety threats without explicit trigger keywords**—for example, a customer saying 'driver turned off the lights and asked if I live alone'. Because our lexical safety rules look for explicit words ('accident', 'drunk', 'police'), situational threats can slip past bag-of-words filters.
Other modes include social media slang/typos ('took my bag of cash n ghosted me'), sparse retrieval on niche policies (emotional support animals), and multi-intent tweets where a customer complains about both a rude driver and a cancellation fee."

---

### Q13: What would you do next with one more week?
**Answer**:
"With one more week:
1. Fine-tune a lightweight contextual transformer (e.g. DistilBERT-safety) to catch implicit harassment and safety threats.
2. Upgrade to multi-label intent classification with a severity hierarchy (safety intent always overrides billing intent).
3. Ingest Uber's official Help Center FAQ pages into our vector index to eliminate retrieval sparsity on niche policies.
4. Implement dynamic confidence thresholds per intent (lower thresholds for routine receipts, strict thresholds for driver conduct)."

---

### Q14: How does your UI differ from standard dashboards?
**Answer**:
"We designed an **Urban Mobility & Fleet Dispatch Operations Console** using a sleek Midnight Onyx (`#0B0F19`), Slate (`#1E293B`), and Electric Cyan (`#06B6D4`) visual language. It features an interactive test console, full evidence inspection with similarity meters, safety badge highlights, and an interactive evaluation analytics suite."

---

### Q15: How fast does the entire benchmark run?
**Answer**:
"The entire pipeline—data preparation, model training, and 200-sample benchmark evaluation—runs in **under 25 seconds** on a standard CPU with zero GPU requirement, easily meeting the assignment's under-15-minute reproduction requirement."
