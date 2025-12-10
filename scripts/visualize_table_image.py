#!/usr/bin/env python3
"""
Visualize table data from image (Tokenization Granularity & Flip Rate)

Creates two plots:
 - Tokenization granularity (avg_tokens_romaji / avg_tokens_native)
 - Flip rate (%) with annotated flip counts ("1->0" and "0->1")

McNemar significance is shown as a star above the bar when True.

Usage:
  python3 scripts/visualize_table_image.py --outdir outputs/figures

This script uses the numeric values shown in the attached image (hardcoded).
"""
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# # romaji-trained models
# DATA = [
#     {
#         "model": "mDeBERTa",
#         "token_ratio": 1.267,
#         "flip_rate_pct": 17.76,
#         "flip_1_to_0": 78,
#         "flip_0_to_1": 54,
#         "mcnemar_significant": True,
#     },
#     {
#         "model": "BERT Japanese",
#         "token_ratio": 1.964,
#         "flip_rate_pct": 22.88,
#         "flip_1_to_0": 94,
#         "flip_0_to_1": 76,
#         "mcnemar_significant": False,
#     },
# ]

# native-trained models
DATA = [
    {
        "model": "mDeBERTa",
        "token_ratio": 1.267,
        "flip_rate_pct": 33.78,
        "flip_1_to_0": 249,
        "flip_0_to_1": 2,
        "mcnemar_significant": True,
    },
    {
        "model": "BERT Japanese",
        "token_ratio": 1.964,
        "flip_rate_pct": 42.12,
        "flip_1_to_0": 313,
        "flip_0_to_1": 0,
        "mcnemar_significant": True,
    },
]


def plot_token_ratio(df, outdir):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    # use same model hues as other plots
    color_map = {
        "mDeBERTa": "#2a92d6",
        "BERT Japanese": "#f0b24a",
    }

    models = df["model"].tolist()
    values = df["token_ratio"].tolist()
    colors = [color_map.get(m, "#6c6c6c") for m in models]

    x = range(len(models))
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Tokenization Granularity\n(romaji / native)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(values) * 1.2)

    # annotate values above bars
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + (max(values) * 0.02),
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    outpath = os.path.join(outdir, "tokenization_granularity.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print("Wrote", outpath)


def plot_flip_rates(df, outdir):
    # Create stacked bar chart where each bar's total equals flip_rate_pct and
    # the two segments correspond to 1->0 and 0->1 flip counts (scaled to percentages).
    sns.set(style="whitegrid")
    models = df["model"].tolist()

    # compute segment percentages per-model
    seg1 = []  # 1->0 percentages
    seg0 = []  # 0->1 percentages
    for _, row in df.iterrows():
        total_flips = row["flip_1_to_0"] + row["flip_0_to_1"]
        if row["flip_rate_pct"] > 0:
            # infer total examples from total_flips and flip_rate_pct
            total_examples = total_flips / (row["flip_rate_pct"] / 100.0)
            # defensive: avoid division by zero
            if total_examples <= 0:
                total_examples = 1.0
        else:
            total_examples = 1.0

        seg1.append(row["flip_1_to_0"] / total_examples * 100.0)
        seg0.append(row["flip_0_to_1"] / total_examples * 100.0)

    x = range(len(models))
    width = 0.6
    # Per-model base colors (same hue, different models)
    base_color_map = {
        "mDeBERTa": "#2a92d6",
        "BERT Japanese": "#f0b24a",
    }

    def shade(hexcolor, factor=0.9):
        h = hexcolor.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = int(max(0, min(255, r * factor)))
        g = int(max(0, min(255, g * factor)))
        b = int(max(0, min(255, b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    # Compute segment heights so that seg1 + seg0 == flip_rate_pct
    seg1 = []
    seg0 = []
    seg1_fracs = []  # fraction within flips for annotation
    seg0_fracs = []
    seg1_colors = []
    seg0_colors = []

    for _, row in df.iterrows():
        f10 = float(row.get("flip_1_to_0", 0))
        f01 = float(row.get("flip_0_to_1", 0))
        total_flips = f10 + f01
        flip_rate = float(row.get("flip_rate_pct", 0.0))
        if total_flips > 0:
            frac1 = f10 / total_flips
            frac0 = f01 / total_flips
        else:
            frac1 = 0.0
            frac0 = 0.0

        # segment heights are proportion of the overall flip_rate
        seg1.append(frac1 * flip_rate)
        seg0.append(frac0 * flip_rate)
        seg1_fracs.append(frac1)
        seg0_fracs.append(frac0)

        base = base_color_map.get(row["model"], "#6c6c6c")
        # Use consistent colors across models: blue for 1->0, orange for 0->1
        # Blue (1->0): darker; Orange (0->1): lighter
        seg1_colors.append("#2a92d6")
        seg0_colors.append("#f0b24a")

    fig, ax = plt.subplots(figsize=(8, 4))
    p1 = ax.bar(x, seg1, width, color=seg1_colors, label="1->0: toxic -> non-toxic")
    p2 = ax.bar(
        x, seg0, width, bottom=seg1, color=seg0_colors, label="0->1: non-toxic -> toxic"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Flip Rate (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(df["flip_rate_pct"]) * 1.25)

    # annotate totals only (no counts inside segments, no McNemar marker)
    # annotate totals and per-segment percentages (as percent of flips)
    for i, row in df.iterrows():
        total_pct = seg1[i] + seg0[i]
        # top label: total flip rate
        ax.text(
            i,
            total_pct + 0.6,
            f"{row['flip_rate_pct']:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        # annotate each segment with percent of flips and absolute count
        total_flips = float(row.get("flip_1_to_0", 0) + row.get("flip_0_to_1", 0))
        if total_flips > 0:
            pct1 = seg1_fracs[i] * 100.0
            pct0 = seg0_fracs[i] * 100.0
            # labels: show percent (of flips) and count
            lbl1 = f"{pct1:.1f}% ({int(row['flip_1_to_0'])})"
            lbl0 = f"{pct0:.1f}% ({int(row['flip_0_to_1'])})"

            # place labels centered in each segment if segment height large enough,
            # otherwise place just above the segment
            y1 = seg1[i] / 2.0
            y0 = seg1[i] + seg0[i] / 2.0

            color1 = "white" if seg1[i] > (max(df["flip_rate_pct"]) * 0.06) else "black"
            color0 = "white" if seg0[i] > (max(df["flip_rate_pct"]) * 0.06) else "black"

            ax.text(i, y1, lbl1, ha="center", va="center", fontsize=9, color=color1)
            ax.text(i, y0, lbl0, ha="center", va="center", fontsize=9, color=color0)

    # place the legend centered above the plot to improve readability
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False)
    # (star legend removed as requested)

    outpath = os.path.join(outdir, "flip_rates_with_counts_stacked.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print("Wrote", outpath)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize table values (tokenization granularity and flip rates)"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Directory to write figures"
    )
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame(DATA)

    plot_token_ratio(df, outdir)
    plot_flip_rates(df, outdir)
    plot_oov_rates(df, outdir)


def plot_oov_rates(df, outdir):
    """Plot OOV (unknown-token) rates for native vs romaji per model.

    If the dataframe does not include `oov_rate_native` / `oov_rate_romaji`,
    fall back to hard-coded values provided by the user.
    """
    sns.set(style="whitegrid")

    # fallback OOV rates (percent values) if not present in DATA
    fallback = {
        "BERT Japanese": {"native": 0.221, "romaji": 0.112},
        "mDeBERTa": {"native": 0.011, "romaji": 0.00659},
    }

    models = df["model"].tolist()
    native_vals = []
    romaji_vals = []
    for _, row in df.iterrows():
        native = row.get("oov_rate_native", None)
        romaji = row.get("oov_rate_romaji", None)
        if native is None or romaji is None:
            fb = fallback.get(row["model"], {"native": 0.0, "romaji": 0.0})
            native = fb["native"]
            romaji = fb["romaji"]
        native_vals.append(float(native))
        romaji_vals.append(float(romaji))

    x = np.arange(len(models))
    width = 0.35

    # model base colors (use same hue mapping as flip plot)
    base_color_map = {
        "mDeBERTa": "#2a92d6",
        "BERT Japanese": "#f0b24a",
    }

    def shade(hexcolor, factor=0.9):
        h = hexcolor.lstrip("#")
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        r = int(max(0, min(255, r * factor)))
        g = int(max(0, min(255, g * factor)))
        b = int(max(0, min(255, b * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    # Use consistent colors: native -> blue, romaji -> orange
    native_colors = ["#2a92d6"] * len(models)
    romaji_colors = ["#f0b24a"] * len(models)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_native = ax.bar(
        x - width / 2, native_vals, width, label="Native (OOV %)", color=native_colors
    )
    bars_romaji = ax.bar(
        x + width / 2, romaji_vals, width, label="Romaji (OOV %)", color=romaji_colors
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("OOV Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("OOV Rates: Native vs Romaji")

    # annotate bars with percent values
    for b in bars_native:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + max(native_vals) * 0.02,
            f"{h:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for b in bars_romaji:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + max(romaji_vals) * 0.02,
            f"{h:.3f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.legend(loc="upper right")
    outpath = os.path.join(outdir, "oov_rates_by_model.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print("Wrote", outpath)


if __name__ == "__main__":
    main()
