#!/usr/bin/env python3
"""
Plot model performance comparison (native vs romaji)

Creates grouped bar charts for Accuracy, Macro F1 and Toxic Recall
and a small line/delta plot showing the change from native -> romaji
for each model.

Usage:
  python3 scripts/plot_model_performance.py --outdir outputs/figures

This script hardcodes the table values provided in the attachments.
It requires matplotlib and seaborn.
"""
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# romaji-trained models
DATA = {
    "Japanese BERT": {
        "native": {"Accuracy": 75, "Macro F1": 73, "Toxic Recall": 63},
        "romaji": {"Accuracy": 78, "Macro F1": 77, "Toxic Recall": 65},
    },
    "mdeberta": {
        "native": {"Accuracy": 79, "Macro F1": 78, "Toxic Recall": 73},
        "romaji": {"Accuracy": 83, "Macro F1": 82, "Toxic Recall": 74},
    },
    "ByT5": {
        "native": {"Accuracy": 61, "Macro F1": 60, "Toxic Recall": 59},
        "romaji": {"Accuracy": 59, "Macro F1": 59, "Toxic Recall": 57},
    },
}

# native-trained models
# DATA = {
#     "Japanese BERT": {
#         "native": {"Accuracy": 93, "Macro F1": 93, "Toxic Recall": 94},
#         "romaji": {"Accuracy": 60, "Macro F1": 38, "Toxic Recall": 0},
#     },
#     "mdeberta": {
#         "native": {"Accuracy": 92, "Macro F1": 91, "Toxic Recall": 91},
#         "romaji": {"Accuracy": 66, "Macro F1": 53, "Toxic Recall": 16},
#     },
#     "ByT5": {
#         "native": {"Accuracy": 61, "Macro F1": 60, "Toxic Recall": 59},
#         "romaji": {"Accuracy": 59, "Macro F1": 59, "Toxic Recall": 57},
#     },
# }


def build_dataframe(data):
    rows = []
    for model, views in data.items():
        for view_name, metrics in views.items():
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "model": model,
                        "view": view_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_grouped_bars(df, outdir):
    sns.set(style="whitegrid")
    metrics = df["metric"].unique()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        # pivot: rows=model, columns=view
        pivot = sub.pivot(index="model", columns="view", values="value")
        pivot = pivot.reindex(index=list(DATA.keys()))
        bars = pivot.plot(kind="bar", ax=ax, rot=0, width=0.75)
        ax.set_title(metric)
        ax.set_ylabel(metric + " (%)")
        ax.set_xlabel("Model")
        ax.legend(title="View")

        # Annotate bar values on top of each bar
        for container in ax.containers:
            # container is a BarContainer
            labels = [int(v.get_height()) for v in container]
            ax.bar_label(
                container, labels=[f"{lab}" for lab in labels], padding=3, fontsize=9
            )

    plt.tight_layout()
    outpath = os.path.join(outdir, "performance_grouped_bars.png")
    fig.savefig(outpath, dpi=200)
    print("Wrote", outpath)


def plot_deltas(df, outdir):
    # create delta (romaji - native) per model per metric
    pivot = df.pivot_table(index=["model", "view"], columns="metric", values="value")
    # pivot has MultiIndex; compute delta
    native = pivot.xs("native", level="view")
    romaji = pivot.xs("romaji", level="view")
    delta = romaji - native

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    # plot lines per metric across models
    metrics = delta.columns.tolist()
    x = range(len(delta.index))
    markers = ["o", "s", "^"]
    for m, marker in zip(metrics, markers):
        ax.plot(list(delta.index), delta[m].values, marker=marker, label=m)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Romaji − Native (percentage points)")
    ax.set_title("Change from Native → Romaji by Model")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    outpath = os.path.join(outdir, "performance_deltas.png")
    fig.savefig(outpath, dpi=200)
    print("Wrote", outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default="outputs/figures", help="output directory for figures"
    )
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = build_dataframe(DATA)
    # ensure metrics ordering
    metric_order = ["Accuracy", "Macro F1", "Toxic Recall"]
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)

    plot_grouped_bars(df, args.outdir)
    plot_deltas(df, args.outdir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
