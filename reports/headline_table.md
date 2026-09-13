# Benchmark Results: Baselines vs Uber AI Support Agent

Evaluated on **200 leak-free golden hand-labelled examples**.

| Metric | Trivial Baseline | Simple Baseline | Uber AI Support Agent |
|---|---|---|---|
| Intent Accuracy | 16.0% | 53.5% | 89.5% |
| Intent Macro F1 | 3.9% | 56.1% | 88.9% |
| Escalation Accuracy | 50.5% | 58.0% | 55.0% |
| False Auto-Handle Rate (Safety Risk) | 100.0% | 33.3% | 0.0% |
| False Escalation Rate | 0.0% | 50.5% | 89.1% |
| Retrieval Top-1 Match | 0.0% | 53.5% | 52.0% |
| Judge Quality Score (1-5) | 3.46 | 4.20 | 4.53 |
| Judge Groundedness (1-5) | 2.00 | 4.60 | 4.11 |
| Unsupported Claims Rate | 0.0% | 0.0% | 0.0% |