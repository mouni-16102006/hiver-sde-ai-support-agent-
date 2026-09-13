"""
Human vs LLM Judge agreement evaluation module.
Calculates Cohen's Kappa, percentage agreement, and Pearson/Spearman correlation
between human annotator ratings and automated judge rubric scores.
"""
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    """
    Computes Cohen's Kappa coefficient for inter-rater agreement on categorical/ordinal scales.
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    categories = sorted(list(set(rater1 + rater2)))
    n = len(rater1)
    k = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Build confusion matrix
    cm = np.zeros((k, k), dtype=int)
    for r1, r2 in zip(rater1, rater2):
        cm[cat_to_idx[r1], cat_to_idx[r2]] += 1

    # Observed agreement
    po = np.trace(cm) / n

    # Chance agreement
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    pe = np.sum(sum_rows * sum_cols) / (n * n)

    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def compute_human_judge_agreement(
    human_scores: List[float],
    judge_scores: List[float],
) -> Dict:
    """
    Computes agreement metrics between human evaluator ratings and automated judge scores:
    - Percentage exact agreement
    - Percentage close agreement (+-1 point on 1-5 scale)
    - Cohen's Kappa (discretized to integer buckets)
    - Mean Absolute Error (MAE)
    """
    if not human_scores or not judge_scores or len(human_scores) != len(judge_scores):
        return {
            "status": "pending_human_annotation",
            "message": "Human evaluation scores are pending or incomplete.",
            "sample_count": 0,
        }

    n = len(human_scores)
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)

    exact_matches = sum(1 for h, j in zip(h_arr, j_arr) if round(h) == round(j))
    close_matches = sum(1 for h, j in zip(h_arr, j_arr) if abs(h - j) <= 1.0)
    mae = float(np.mean(np.abs(h_arr - j_arr)))

    h_int = [int(round(x)) for x in h_arr]
    j_int = [int(round(x)) for x in j_arr]
    kappa = compute_cohens_kappa(h_int, j_int)

    return {
        "status": "evaluated",
        "sample_count": n,
        "exact_agreement_rate": round(exact_matches / n, 4),
        "close_agreement_rate": round(close_matches / n, 4),
        "mean_absolute_error": round(mae, 4),
        "cohens_kappa": kappa,
        "interpretation": (
            "Substantial agreement" if kappa >= 0.61 else
            "Moderate agreement" if kappa >= 0.41 else
            "Fair agreement" if kappa >= 0.21 else "Slight agreement"
        ),
    }
