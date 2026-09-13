import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Evaluation execution script for Uber AI Support Agent.
Runs the evaluation harness on the Golden Evaluation Set, generates headline comparisons,
and formats markdown tables for reporting.
"""
import json
import logging
import pandas as pd
from evaluation.evaluate import run_full_evaluation
from evaluation.human_agreement import compute_human_judge_agreement
from src.config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Executing full benchmark evaluation...")
    results = run_full_evaluation()

    headline = results["headline_comparison"]
    df_headline = pd.DataFrame(headline)
    df_headline.columns = ["Metric", "Trivial Baseline", "Simple Baseline", "Uber AI Support Agent"]

    print("\n" + "=" * 80)
    print("UBER AI SUPPORT AGENT — HEADLINE EVALUATION RESULTS")
    print("=" * 80)
    print(df_headline.to_string(index=False))
    print("=" * 80 + "\n")

    # Save markdown summary
    md_lines = [
        "# Benchmark Results: Baselines vs Uber AI Support Agent\n",
        f"Evaluated on **{results['dataset_size']} leak-free golden hand-labelled examples**.\n",
        "| Metric | Trivial Baseline | Simple Baseline | Uber AI Support Agent |",
        "|---|---|---|---|",
    ]
    for row in df_headline.itertuples(index=False):
        md_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    md_content = "\n".join(md_lines)
    (REPORTS_DIR / "headline_table.md").write_text(md_content, encoding="utf-8")
    logger.info(f"Saved headline table markdown to {REPORTS_DIR / 'headline_table.md'}")

    # Compute human agreement demonstration on sample rated subset
    judge_scores = [rec["judge_evaluation"]["overall_score"] for rec in results["models"]["uber_ai_agent"]["detailed_intent"].get("records", [])]
    # Synthetic dual ratings for template testing if human ratings pending
    human_sample_csv = Path(__file__).resolve().parent.parent / "data" / "golden" / "human_sample_annotations.csv"
    if human_sample_csv.exists():
        df_human = pd.read_csv(human_sample_csv)
        h_scores = df_human["human_overall_score"].dropna().tolist()
        j_scores = df_human["judge_overall_score"].dropna().tolist()
        agreement = compute_human_judge_agreement(h_scores, j_scores)
        with open(REPORTS_DIR / "human_agreement_results.json", "w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2)
        logger.info(f"Saved human agreement metrics to {REPORTS_DIR / 'human_agreement_results.json'}")


if __name__ == "__main__":
    main()
