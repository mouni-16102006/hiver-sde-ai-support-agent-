"""
Evaluation metrics calculation module for Uber AI Support Agent.
Computes intent metrics, escalation metrics, retrieval metrics, and response quality.
"""
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.config import DECISION_AUTO_HANDLE, DECISION_ESCALATE, INTENTS


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Computes multi-class classification metrics for intent prediction.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-intent metrics
    labels = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_intent": {k: report[k] for k in labels if k in report},
        "labels": labels,
        "confusion_matrix": cm,
    }


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Computes binary evaluation metrics for auto-handle vs escalate decision.
    Specifically measures critical safety risk: False Auto-Handle Rate.
    """
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, pos_label=DECISION_ESCALATE, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=DECISION_ESCALATE, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=DECISION_ESCALATE, zero_division=0)

    # Safety-critical rate: Should escalate, but wrongly auto-handled
    total_should_escalate = sum(1 for yt in y_true if yt == DECISION_ESCALATE)
    false_auto_handles = sum(
        1 for yt, yp in zip(y_true, y_pred)
        if yt == DECISION_ESCALATE and yp == DECISION_AUTO_HANDLE
    )
    false_auto_handle_rate = (false_auto_handles / total_should_escalate) if total_should_escalate > 0 else 0.0

    # Unnecessary escalations: Routine inquiry wrongly escalated to human
    total_should_auto = sum(1 for yt in y_true if yt == DECISION_AUTO_HANDLE)
    false_escalations = sum(
        1 for yt, yp in zip(y_true, y_pred)
        if yt == DECISION_AUTO_HANDLE and yp == DECISION_ESCALATE
    )
    false_escalation_rate = (false_escalations / total_should_auto) if total_should_auto > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "false_auto_handles_count": false_auto_handles,
        "false_escalation_rate": round(float(false_escalation_rate), 4),
        "false_escalations_count": false_escalations,
    }


def compute_retrieval_metrics(retrieval_results: List[List[Dict]], gold_intents: List[str]) -> Dict:
    """
    Computes retrieval accuracy and top-1 intent alignment.
    """
    top1_intent_matches = 0
    sim_scores = []

    for res, gold in zip(retrieval_results, gold_intents):
        if res:
            sim_scores.append(res[0].get("similarity", 0.0))
            if res[0].get("intent") == gold:
                top1_intent_matches += 1

    total = len(gold_intents)
    top1_intent_acc = (top1_intent_matches / total) if total > 0 else 0.0
    mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0

    return {
        "top1_intent_match_rate": round(top1_intent_acc, 4),
        "mean_top1_similarity": round(mean_sim, 4),
        "total_evaluated": total,
    }
