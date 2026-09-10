"""
evaluate.py: Quantitative evaluation of the Explainable Skill Extraction and Matching System.

Calculates authentic, un-fabricated metrics:
- True Positives (TP), False Positives (FP), False Negatives (FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- Jaccard Similarity / Match Accuracy: TP / (TP + FP + FN)

Evaluates on data/evaluation_set.json.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from src.skill_extractor import SkillExtractor
from src.skill_normalizer import SkillNormalizer
from src.config import BASE_DIR


EVALUATION_DATA_PATH = BASE_DIR / "data" / "evaluation_set.json"


def load_evaluation_dataset(path: Path = EVALUATION_DATA_PATH) -> List[Dict[str, Any]]:
    """Load manually annotated ground-truth test cases."""
    if not path.exists():
        print(f"\n[INFO] Evaluation dataset not found at: {path}")
        print("To create an evaluation dataset, save a JSON file with the structure:")
        print("""
[
  {
    "id": "sample_1",
    "domain": "Data Science",
    "text": "Resume or job description text here...",
    "ground_truth_skills": ["python", "machine learning", "sql"]
  }
]
        """)
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_system():
    """Run evaluation and report genuine Precision, Recall, and F1-Score."""
    print("=" * 65)
    print("SKILL GAP ANALYZER - QUANTITATIVE EVALUATION SUITE")
    print("=" * 65)

    dataset = load_evaluation_dataset()
    if not dataset:
        print("\nNo evaluation samples available. Exiting evaluation without fake metrics.")
        return

    normalizer = SkillNormalizer()
    extractor = SkillExtractor(normalizer=normalizer)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    sample_results = []

    print(f"\nEvaluating on {len(dataset)} annotated benchmark test cases:\n")
    print(f"{'Sample ID':<15} | {'Domain':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for item in dataset:
        sample_id = item.get("id", "Unknown")
        domain = item.get("domain", "General")
        text = item.get("text", "")
        # Normalize ground truth skills
        gt_skills = set(normalizer.normalize_skill_list(item.get("ground_truth_skills", [])))

        # Extract predictions
        predicted_skills = set(extractor.extract_skills(text))

        # Calculate TP, FP, FN
        tp_set = predicted_skills.intersection(gt_skills)
        fp_set = predicted_skills.difference(gt_skills)
        fn_set = gt_skills.difference(predicted_skills)

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        sample_results.append({
            "id": sample_id,
            "domain": domain,
            "ground_truth": sorted(list(gt_skills)),
            "predicted": sorted(list(predicted_skills)),
            "tp": sorted(list(tp_set)),
            "fp": sorted(list(fp_set)),
            "fn": sorted(list(fn_set)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        print(f"{sample_id:<15} | {domain[:22]:<22} | {precision * 100:>8.2f}% | {recall * 100:>8.2f}% | {f1 * 100:>8.2f}%")

    # Global Aggregate Metrics (Micro-Averaged)
    global_precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    global_recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = (
        (2 * global_precision * global_recall / (global_precision + global_recall))
        if (global_precision + global_recall) > 0
        else 0.0
    )
    jaccard_accuracy = (
        (total_tp / (total_tp + total_fp + total_fn))
        if (total_tp + total_fp + total_fn) > 0
        else 0.0
    )

    print("-" * 75)
    print("\nOVERALL BENCHMARK RESULTS (Micro-Averaged):")
    print(f"  • Total Ground Truth Skills: {total_tp + total_fn}")
    print(f"  • Correctly Extracted (TP) : {total_tp}")
    print(f"  • False Positives      (FP) : {total_fp}")
    print(f"  • Missed Skills        (FN) : {total_fn}")
    print(f"  • Precision                 : {global_precision * 100:.2f}%")
    print(f"  • Recall                    : {global_recall * 100:.2f}%")
    print(f"  • F1-Score                  : {global_f1 * 100:.2f}%")
    print(f"  • Jaccard Match Accuracy    : {jaccard_accuracy * 100:.2f}%")
    print("=" * 65)

    # Detailed Sample Breakdown
    print("\nDetailed Error Analysis for Inspection:")
    for r in sample_results:
        print(f"\n--- [{r['id']}] {r['domain']} ---")
        print(f"  Matched (TP): {', '.join(r['tp'])}")
        if r['fp']:
            print(f"  False Positives (FP): {', '.join(r['fp'])}")
        else:
            print("  False Positives (FP): None (0)")
        if r['fn']:
            print(f"  False Negatives (FN): {', '.join(r['fn'])}")
        else:
            print("  False Negatives (FN): None (0)")


if __name__ == "__main__":
    evaluate_system()
