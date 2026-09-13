"""
Main evaluation harness runner for Uber AI Support Agent.
Benchmarks Proposed Agent vs Simple Baseline vs Trivial Baseline
on the Golden Evaluation Set with zero data leakage.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from baselines.simple_baseline import SimpleBaselineAgent
from baselines.trivial import TrivialBaselineAgent
from evaluation.error_analysis import extract_top_failure_modes
from evaluation.judge import SupportJudge
from evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    compute_retrieval_metrics,
)
from src.agent import UberAIAgent
from src.config import (
    EVAL_RESULTS_JSON,
    GOLDEN_EVAL_FILE,
    HISTORICAL_KB_FILE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def run_full_evaluation(
    golden_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Runs end-to-end evaluation across all three systems on the golden evaluation set.
    """
    target_golden = golden_path or GOLDEN_EVAL_FILE
    if not target_golden.exists():
        raise FileNotFoundError(f"Golden evaluation set not found at {target_golden}")

    with open(target_golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    logger.info(f"Loaded {len(golden_set)} golden evaluation cases.")

    # Instantiate agents
    agent = UberAIAgent()
    trivial_agent = TrivialBaselineAgent()

    # Simple baseline uses historical kb
    kb_docs = []
    if HISTORICAL_KB_FILE.exists():
        with open(HISTORICAL_KB_FILE, "r", encoding="utf-8") as f:
            kb_docs = json.load(f)
    simple_agent = SimpleBaselineAgent(documents=kb_docs)

    judge = SupportJudge()

    # Storage for predictions
    gold_intents = [g["gold_intent"] for g in golden_set]
    gold_decisions = [g["gold_decision"] for g in golden_set]
    test_queries = [g["customer_message"] for g in golden_set]

    agent_records = []
    simple_records = []
    trivial_records = []

    for item in golden_set:
        msg = item["customer_message"]
        g_intent = item["gold_intent"]
        g_decision = item["gold_decision"]

        # Run Proposed Agent (preventing self-retrieval leakage)
        res_agent = agent.process_message(msg, exclude_texts=[msg])
        j_agent = judge.evaluate_response(
            customer_message=msg,
            gold_intent=g_intent,
            gold_decision=g_decision,
            predicted_intent=res_agent["intent"],
            predicted_decision=res_agent["decision"],
            generated_reply=res_agent["reply"],
            evidence=res_agent["evidence"],
        )
        res_agent.update({
            "gold_intent": g_intent,
            "gold_decision": g_decision,
            "judge_evaluation": j_agent,
        })
        agent_records.append(res_agent)

        # Run Simple Baseline
        res_simple = simple_agent.process_message(msg, exclude_texts=[msg])
        j_simple = judge.evaluate_response(
            customer_message=msg,
            gold_intent=g_intent,
            gold_decision=g_decision,
            predicted_intent=res_simple["intent"],
            predicted_decision=res_simple["decision"],
            generated_reply=res_simple["reply"],
            evidence=res_simple["evidence"],
        )
        res_simple.update({
            "gold_intent": g_intent,
            "gold_decision": g_decision,
            "judge_evaluation": j_simple,
        })
        simple_records.append(res_simple)

        # Run Trivial Baseline
        res_trivial = trivial_agent.process_message(msg)
        j_trivial = judge.evaluate_response(
            customer_message=msg,
            gold_intent=g_intent,
            gold_decision=g_decision,
            predicted_intent=res_trivial["intent"],
            predicted_decision=res_trivial["decision"],
            generated_reply=res_trivial["reply"],
            evidence=[],
        )
        res_trivial.update({
            "gold_intent": g_intent,
            "gold_decision": g_decision,
            "judge_evaluation": j_trivial,
        })
        trivial_records.append(res_trivial)

    # Compute aggregate metrics for all 3 models
    def get_model_summary(records: List[Dict]) -> Dict:
        p_intents = [r["intent"] for r in records]
        p_decisions = [r["decision"] for r in records]
        intent_m = compute_intent_metrics(gold_intents, p_intents)
        escalation_m = compute_escalation_metrics(gold_decisions, p_decisions)
        retrieval_m = compute_retrieval_metrics([r.get("evidence", []) for r in records], gold_intents)

        # Judge averages
        judge_scores = [r["judge_evaluation"]["overall_score"] for r in records]
        groundedness_scores = [r["judge_evaluation"]["groundedness"] for r in records]
        relevance_scores = [r["judge_evaluation"]["relevance"] for r in records]
        unsupported_count = sum(1 for r in records if r["judge_evaluation"]["has_unsupported_claim"])

        return {
            "intent_accuracy": intent_m["accuracy"],
            "intent_macro_f1": intent_m["macro_f1"],
            "escalation_accuracy": escalation_m["accuracy"],
            "escalation_precision": escalation_m["precision"],
            "escalation_recall": escalation_m["recall"],
            "false_auto_handle_rate": escalation_m["false_auto_handle_rate"],
            "false_escalation_rate": escalation_m["false_escalation_rate"],
            "retrieval_top1_match": retrieval_m["top1_intent_match_rate"],
            "mean_top1_similarity": retrieval_m["mean_top1_similarity"],
            "judge_mean_quality": round(float(pd.Series(judge_scores).mean()), 2),
            "judge_mean_groundedness": round(float(pd.Series(groundedness_scores).mean()), 2),
            "judge_mean_relevance": round(float(pd.Series(relevance_scores).mean()), 2),
            "unsupported_claim_rate": round(unsupported_count / len(records), 4),
            "detailed_intent": intent_m,
            "detailed_escalation": escalation_m,
        }

    summary_agent = get_model_summary(agent_records)
    summary_simple = get_model_summary(simple_records)
    summary_trivial = get_model_summary(trivial_records)

    # Error analysis on proposed agent
    top_failures = extract_top_failure_modes(agent_records)

    comparison_results = {
        "dataset_size": len(golden_set),
        "headline_comparison": {
            "metric": [
                "Intent Accuracy",
                "Intent Macro F1",
                "Escalation Accuracy",
                "False Auto-Handle Rate (Safety Risk)",
                "False Escalation Rate",
                "Retrieval Top-1 Match",
                "Judge Quality Score (1-5)",
                "Judge Groundedness (1-5)",
                "Unsupported Claims Rate",
            ],
            "trivial_baseline": [
                f"{summary_trivial['intent_accuracy'] * 100:.1f}%",
                f"{summary_trivial['intent_macro_f1'] * 100:.1f}%",
                f"{summary_trivial['escalation_accuracy'] * 100:.1f}%",
                f"{summary_trivial['false_auto_handle_rate'] * 100:.1f}%",
                f"{summary_trivial['false_escalation_rate'] * 100:.1f}%",
                "0.0%",
                f"{summary_trivial['judge_mean_quality']:.2f}",
                f"{summary_trivial['judge_mean_groundedness']:.2f}",
                f"{summary_trivial['unsupported_claim_rate'] * 100:.1f}%",
            ],
            "simple_baseline": [
                f"{summary_simple['intent_accuracy'] * 100:.1f}%",
                f"{summary_simple['intent_macro_f1'] * 100:.1f}%",
                f"{summary_simple['escalation_accuracy'] * 100:.1f}%",
                f"{summary_simple['false_auto_handle_rate'] * 100:.1f}%",
                f"{summary_simple['false_escalation_rate'] * 100:.1f}%",
                f"{summary_simple['retrieval_top1_match'] * 100:.1f}%",
                f"{summary_simple['judge_mean_quality']:.2f}",
                f"{summary_simple['judge_mean_groundedness']:.2f}",
                f"{summary_simple['unsupported_claim_rate'] * 100:.1f}%",
            ],
            "uber_ai_agent": [
                f"{summary_agent['intent_accuracy'] * 100:.1f}%",
                f"{summary_agent['intent_macro_f1'] * 100:.1f}%",
                f"{summary_agent['escalation_accuracy'] * 100:.1f}%",
                f"{summary_agent['false_auto_handle_rate'] * 100:.1f}%",
                f"{summary_agent['false_escalation_rate'] * 100:.1f}%",
                f"{summary_agent['retrieval_top1_match'] * 100:.1f}%",
                f"{summary_agent['judge_mean_quality']:.2f}",
                f"{summary_agent['judge_mean_groundedness']:.2f}",
                f"{summary_agent['unsupported_claim_rate'] * 100:.1f}%",
            ],
        },
        "models": {
            "uber_ai_agent": summary_agent,
            "simple_baseline": summary_simple,
            "trivial_baseline": summary_trivial,
        },
        "top_failure_modes": top_failures,
    }

    target_out = output_path or EVAL_RESULTS_JSON
    target_out.parent.mkdir(parents=True, exist_ok=True)
    with open(target_out, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # Save headline comparison separately
    with open(REPORTS_DIR / "headline_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results["headline_comparison"], f, indent=2)

    logger.info(f"Evaluation finished. Results saved to {target_out}")
    return comparison_results
