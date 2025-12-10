#!/usr/bin/env python3
"""
Compute Error Taxonomy (Type B / Type C ratio) for Section 7.2.3 analysis.

This script analyzes model prediction results to categorize errors into:
- Type A: Model correct (baseline - can handle this case)
- Type B: Model fails on toxic content (missed toxicity - false negative)
- Type C: Model fails on non-toxic content (false alarm - false positive)
- Type D: Not applicable for single model analysis

Type B / Type C ratio helps understand if the model is more prone to:
- Missing toxic content (high Type B) vs
- Over-flagging non-toxic content (high Type C)

Usage:
    python compute_error_taxonomy.py outputs/eval/bert_romaji_results.csv
    python compute_error_taxonomy.py outputs/eval/mdeberta_romaji_results.csv
    python compute_error_taxonomy.py outputs/eval/bert_romaji_results.csv outputs/eval/mdeberta_romaji_results.csv
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


def analyze_single_model(csv_path: str, model_name: str = None):
    """
    Analyze error patterns for a single model.

    Type A: Correct predictions (IsTruePositive = True)
    Type B: False Negatives (predicted non-toxic, actually toxic)
            - ToxicityGT = 1, Results = 0
    Type C: False Positives (predicted toxic, actually non-toxic)
            - ToxicityGT = 0, Results = 1
    Type D: True Negatives (correct non-toxic) - counted in Type A

    Args:
        csv_path: Path to results CSV file
        model_name: Optional model name for display

    Returns:
        Dictionary with error statistics
    """
    if model_name is None:
        model_name = Path(csv_path).stem

    print("=" * 80)
    print(f"ANALYSIS: {model_name}")
    print(f"File: {csv_path}")
    print("=" * 80)

    # Load results
    df = pd.read_csv(csv_path)

    total = len(df)
    print(f"\nTotal samples: {total}")

    # Type A: Correct predictions
    type_a = df[df["IsTruePositive"] == True]
    type_a_count = len(type_a)

    # Type B: False Negatives (model says 0, truth is 1)
    # These are TOXIC posts that the model MISSED
    type_b = df[(df["ToxicityGT"] == 1) & (df["Results"] == 0)]
    type_b_count = len(type_b)

    # Type C: False Positives (model says 1, truth is 0)
    # These are NON-TOXIC posts that the model INCORRECTLY flagged
    type_c = df[(df["ToxicityGT"] == 0) & (df["Results"] == 1)]
    type_c_count = len(type_c)

    # Verify counts add up
    incorrect = type_b_count + type_c_count
    assert type_a_count + incorrect == total, "Count mismatch!"

    # Calculate metrics
    accuracy = type_a_count / total

    print(f"\n{'Category':<20} {'Count':>10} {'Percentage':>12}")
    print("-" * 44)
    print(
        f"{'Type A (Correct)':<20} {type_a_count:>10} {type_a_count/total*100:>11.2f}%"
    )
    print(
        f"{'Type B (False Neg)':<20} {type_b_count:>10} {type_b_count/total*100:>11.2f}%"
    )
    print(
        f"{'Type C (False Pos)':<20} {type_c_count:>10} {type_c_count/total*100:>11.2f}%"
    )
    print("-" * 44)
    print(f"{'Total Errors':<20} {incorrect:>10} {incorrect/total*100:>11.2f}%")
    print(f"{'Accuracy':<20} {type_a_count:>10} {accuracy*100:>11.2f}%")

    # Compute Type B / Type C ratio
    if type_c_count > 0:
        bc_ratio = type_b_count / type_c_count
        print(f"\n{'TYPE B / TYPE C RATIO':<30} {bc_ratio:.4f}")
        print("-" * 80)

        if bc_ratio > 1.0:
            print("⚠️  INTERPRETATION: Ratio > 1.0")
            print(
                f"    Model misses toxic content MORE than it over-flags non-toxic content"
            )
            print(f"    → Tends to under-predict toxicity (conservative)")
            print(
                f"    → {type_b_count} toxic posts missed vs {type_c_count} false alarms"
            )
        elif bc_ratio < 1.0:
            print("⚠️  INTERPRETATION: Ratio < 1.0")
            print(
                f"    Model over-flags non-toxic content MORE than it misses toxic content"
            )
            print(f"    → Tends to over-predict toxicity (aggressive)")
            print(
                f"    → {type_c_count} false alarms vs {type_b_count} toxic posts missed"
            )
        else:
            print("✓  INTERPRETATION: Ratio = 1.0")
            print(f"    Model has balanced error distribution")
            print(f"    → Equal false negatives and false positives")
    else:
        bc_ratio = float("inf") if type_b_count > 0 else 0.0
        print(
            f"\n{'TYPE B / TYPE C RATIO':<30} {'∞' if type_b_count > 0 else '0'} (no false positives)"
        )
        print("-" * 80)
        print("⚠️  INTERPRETATION: No false positives detected")
        print("    Model never over-flags non-toxic content")
        if type_b_count > 0:
            print(f"    But it misses {type_b_count} toxic posts")

    # Additional insights
    print("\n" + "=" * 80)
    print("ADDITIONAL INSIGHTS")
    print("=" * 80)

    # Ground truth distribution
    toxic_count = len(df[df["ToxicityGT"] == 1])
    non_toxic_count = len(df[df["ToxicityGT"] == 0])

    print(f"\nGround Truth Distribution:")
    print(f"  Toxic samples:     {toxic_count:>6} ({toxic_count/total*100:>5.2f}%)")
    print(
        f"  Non-toxic samples: {non_toxic_count:>6} ({non_toxic_count/total*100:>5.2f}%)"
    )

    # Model's prediction distribution
    pred_toxic = len(df[df["Results"] == 1])
    pred_non_toxic = len(df[df["Results"] == 0])

    print(f"\nModel Predictions:")
    print(f"  Predicted toxic:     {pred_toxic:>6} ({pred_toxic/total*100:>5.2f}%)")
    print(
        f"  Predicted non-toxic: {pred_non_toxic:>6} ({pred_non_toxic/total*100:>5.2f}%)"
    )

    # Performance on toxic vs non-toxic
    if toxic_count > 0:
        toxic_recall = (toxic_count - type_b_count) / toxic_count
        print(
            f"\nRecall (on toxic):    {toxic_recall*100:.2f}% ({toxic_count - type_b_count}/{toxic_count} correctly identified)"
        )

    if non_toxic_count > 0:
        specificity = (non_toxic_count - type_c_count) / non_toxic_count
        print(
            f"Specificity (on non-toxic): {specificity*100:.2f}% ({non_toxic_count - type_c_count}/{non_toxic_count} correctly identified)"
        )

    print("\n")

    return {
        "model_name": model_name,
        "total": total,
        "type_a": type_a_count,
        "type_b": type_b_count,
        "type_c": type_c_count,
        "bc_ratio": bc_ratio,
        "accuracy": accuracy,
        "toxic_count": toxic_count,
        "non_toxic_count": non_toxic_count,
    }


def compare_models(results_list):
    """Compare multiple models side by side."""
    if len(results_list) < 2:
        return

    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    print(f"\n{'Metric':<25}", end="")
    for res in results_list:
        print(f"{res['model_name']:<25}", end="")
    print()
    print("-" * (25 + 25 * len(results_list)))

    # Accuracy
    print(f"{'Accuracy':<25}", end="")
    for res in results_list:
        print(f"{res['accuracy']*100:>6.2f}%{'':<18}", end="")
    print()

    # Type B / Type C ratio
    print(f"{'Type B / Type C Ratio':<25}", end="")
    for res in results_list:
        if res["bc_ratio"] == float("inf"):
            print(f"{'∞':<25}", end="")
        else:
            print(f"{res['bc_ratio']:>6.4f}{'':<18}", end="")
    print()

    # False Negatives (Type B)
    print(f"{'Missed Toxicity (False Negatives)':<25}", end="")
    for res in results_list:
        pct = res["type_b"] / res["total"] * 100
        print(f"{res['type_b']:>6} ({pct:>5.2f}%){'':<9}", end="")
    print()

    # False Positives (Type C)
    print(f"{'Over-Flagged Non-Toxic (False Positives)':<25}", end="")
    for res in results_list:
        pct = res["type_c"] / res["total"] * 100
        print(f"{res['type_c']:>6} ({pct:>5.2f}%){'':<9}", end="")
    print()

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute Error Taxonomy (Type B / Type C ratio) for Section 7.2.3"
    )
    parser.add_argument(
        "csv_files", nargs="+", help="One or more CSV result files to analyze"
    )
    parser.add_argument(
        "--output", type=str, help="Optional: Save summary to JSON file"
    )

    args = parser.parse_args()

    # Analyze each file
    results = []
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}", file=sys.stderr)
            continue

        result = analyze_single_model(csv_file)
        results.append(result)

    # Compare if multiple files
    if len(results) > 1:
        compare_models(results)

    # Save summary if requested
    if args.output and results:
        import json

        summary = {"models": results, "timestamp": pd.Timestamp.now().isoformat()}

        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✓ Summary saved to {args.output}")

    print("=" * 80)
    print("SECTION 7.2.3 INTERPRETATION GUIDE")
    print("=" * 80)
    print(
        """
For Section 7.2.3, the Type B / Type C ratio indicates:

• Type B (False Negatives): Model MISSES toxic content
  - Predicted as non-toxic, but actually toxic
  - Harmful because toxic content gets through

• Type C (False Positives): Model OVER-FLAGS non-toxic content
  - Predicted as toxic, but actually non-toxic
  - Annoying but less harmful (false alarms)

Ratio Interpretation:
• Ratio > 1.0: Model is conservative (misses more toxic than over-flags)
  → Suggests potential tokenization issues preventing toxic detection
  → For romaji models: May indicate vocabulary/tokenization problems

• Ratio < 1.0: Model is aggressive (over-flags more than misses)
  → May be too sensitive
  → Could indicate overfitting to certain patterns

• Ratio ≈ 1.0: Balanced error distribution
  → Errors are evenly split between false negatives and false positives
    """
    )


if __name__ == "__main__":
    main()
