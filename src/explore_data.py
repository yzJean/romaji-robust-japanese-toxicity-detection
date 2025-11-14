#!/usr/bin/env python3
"""
Quick data exploration script to understand dataset size and distribution.
"""

import pandas as pd
import numpy as np


def explore_data(csv_path="data/processed/paired_native_romaji_inspection_ai_binary.csv"):
    """Explore the dataset to understand size and distribution."""

    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv(csv_path)

    print(f"📊 Total samples: {len(df)}")
    print(f"📄 Columns: {list(df.columns)}")

    # Label distribution
    print(f"\n🏷️ Label distribution:")
    label_counts = df["label_int_coarse"].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Non-Toxic" if label == 0 else "Toxic"
        percentage = (count / len(df)) * 100
        print(f"  {label} ({label_name}): {count} samples ({percentage:.1f}%)")

    # Text length analysis
    print(f"\n📝 Text length analysis (native Japanese):")
    text_lengths = df["text_native"].str.len()
    print(f"  Min length: {text_lengths.min()}")
    print(f"  Max length: {text_lengths.max()}")
    print(f"  Average length: {text_lengths.mean():.1f}")
    print(f"  Median length: {text_lengths.median():.1f}")

    # Sample texts
    print(f"\n📖 Sample texts:")
    print("Non-Toxic examples:")
    non_toxic = df[df["label_int_coarse"] == 0].head(3)
    for _, row in non_toxic.iterrows():
        print(f"  • {row['text_native']}")

    print("Toxic examples:")
    toxic = df[df["label_int_coarse"] == 1].head(3)
    for _, row in toxic.iterrows():
        print(f"  • {row['text_native']}")

    # Recommendations for quick testing
    print(f"\n💡 RECOMMENDATIONS for quick verification:")
    print(f"  • Ultra-quick test: --quick-test (50 samples)")
    print(f"  • Small test: --sample-size 100 (100 samples)")
    print(f"  • Medium test: --sample-size 200 (200 samples)")
    print(f"  • Large test: --sample-size 500 (500+ samples)")
    print(f"  • Full dataset: {len(df)} samples (may take 10-20 minutes)")

    print(f"\n⚡ Expected training times:")
    print(f"  • 50 samples: ~1-2 minutes")
    print(f"  • 100 samples: ~2-3 minutes")
    print(f"  • 200 samples: ~3-5 minutes")
    print(f"  • Full dataset: ~10-20 minutes")


if __name__ == "__main__":
    import sys

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/processed/paired_native_romaji_inspection_ai_binary.csv"
    )
    explore_data(csv_path)
