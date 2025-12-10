import matplotlib.pyplot as plt
import numpy as np

models = ["Japanese_BERT", "mDeBERTa", "ByT5"]

#### a) Grouped bars for FN% and FP% (two panels) ####

# Example numbers from your two tables:
# native-trained evaluated on romaji (FN%, FP%)
fn_native_tr = [39.84, 33.38, 16.44]
fp_native_tr = [0.00, 1.08, 24.26]

# romaji-trained evaluated on romaji
fn_romaji_tr = [14.13, 10.23, 16.44]
fp_romaji_tr = [7.40, 6.59, 24.26]

# Accuracy values extracted from your tables (native-trained on romaji vs romaji-trained on romaji)
acc_native_tr = [60.16, 65.55, 59.30]  # percent
acc_romaji_tr = [78.47, 83.18, 59.30]  # percent

x = np.arange(len(models))
width = 0.35

# Create output directory if missing
import os

out_dir = "outputs/figures/error-tax"
os.makedirs(out_dir, exist_ok=True)


def annotate_bars(
    bar_containers, fmt="{:.1f}", suffix="%", offset=1.0, inside=False, inside_frac=0.15
):
    """Annotate bar containers with their heights.

    bar_containers can be a single container or an iterable of containers.
    If inside=True, place label inside the bar at height*(1-inside_frac).
    """
    if not hasattr(bar_containers, "__iter__"):
        bar_containers = [bar_containers]
    for container in bar_containers:
        for rect in container:
            h = rect.get_height()
            if np.isfinite(h):
                label = fmt.format(h) + suffix
            else:
                label = "∞"
            ax = rect.axes
            if inside and np.isfinite(h) and h > 0:
                # place label inside the bar
                y = h * (1.0 - inside_frac)
                va = "center"
                color = "black"
            else:
                # place label above the bar
                y = h + offset
                va = "bottom"
                color = "black"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y,
                label,
                ha="center",
                va=va,
                fontsize=8,
                color=color,
            )


# 3-panel figure: Accuracy | False Negatives | False Positives
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

# Accuracy panel
bars_acc_a = axes[0].bar(
    x - width / 2, acc_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_acc_b = axes[0].bar(
    x + width / 2, acc_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_title("Accuracy on Romaji Test")
axes[0].set_ylabel("Percent (%)")
axes[0].legend()

# False Negatives panel
bars_fn_a = axes[1].bar(
    x - width / 2, fn_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_fn_b = axes[1].bar(
    x + width / 2, fn_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_title("Missed Toxicity (False Negatives %) on Romaji")
axes[1].legend()

# False Positives panel
bars_fp_a = axes[2].bar(
    x - width / 2, fp_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_fp_b = axes[2].bar(
    x + width / 2, fp_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
axes[2].set_xticks(x)
axes[2].set_xticklabels(models)
axes[2].set_title("Over-Flagged Non-Toxic (False Positives %) on Romaji")
axes[2].legend()

plt.tight_layout()

# Add numeric labels on each bar (accuracy, FN, FP) BEFORE saving
annotate_bars([bars_acc_a, bars_acc_b], fmt="{:.2f}")
annotate_bars([bars_fn_a, bars_fn_b], fmt="{:.2f}")
annotate_bars([bars_fp_a, bars_fp_b], fmt="{:.2f}")

plt.savefig(os.path.join(out_dir, "fn_fp_acc_comparison_romaji.png"), dpi=200)
# plt.show()

# (annotations added above using the earlier annotate_bars definition)


#### b) Type B / Type C ratio (grouped, log scale; native-trained vs romaji-trained) ####
# Ratios for native-trained (from earlier analysis)
models_short = ["BERT", "mDeBERTa", "ByT5"]
ratios_native = [float("inf"), 31.0, 0.6778]
# Ratios for romaji-trained (from earlier analysis / screenshots)
ratios_romaji = [1.9091, 1.5510, 0.6778]

# Plot grouped bars (two bars per model) on log scale
x2 = np.arange(len(models_short))
width2 = 0.35
fig2, ax2 = plt.subplots(figsize=(8, 4))

# Prepare plotting values with a cap for infinite values
cap = 35.0
plot_native = [r if np.isfinite(r) else cap for r in ratios_native]
plot_romaji = [r if np.isfinite(r) else cap for r in ratios_romaji]

bars_n = ax2.bar(
    x2 - width2 / 2, plot_native, width2, label="native-trained", color="#2a92d6"
)
bars_r = ax2.bar(
    x2 + width2 / 2, plot_romaji, width2, label="romaji-trained", color="#f0b24a"
)

ax2.set_xticks(x2)
ax2.set_xticklabels(models_short)
ax2.set_ylabel("Type B / Type C Ratio")
ax2.set_title("Type B / Type C Ratio — Native vs Romaji-trained (evaluated on Romaji)")
ax2.legend()

# Reference line at 1 (equal Type B and Type C)
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)

# Annotate bars with actual labels (use '∞' for true infinities)
for r_val, rect in zip(ratios_native, bars_n):
    if not np.isfinite(r_val):
        label = "∞"
        # place the infinity symbol close to the top of the capped bar
        y = rect.get_height() * 0.98
        va = "center"
        color = "black"
    else:
        label = f"{r_val:.2f}"
        # place label slightly inside the top of the bar for bette\r visual
        y = rect.get_height() * 0.92
        va = "center"
        color = "black"
    ax2.text(
        rect.get_x() + rect.get_width() / 2,
        y,
        label,
        ha="center",
        va=va,
        fontsize=8,
        color=color,
    )

for r_val, rect in zip(ratios_romaji, bars_r):
    if not np.isfinite(r_val):
        label = "∞"
        y = rect.get_height() * 0.98
        va = "center"
        color = "black"
    else:
        label = f"{r_val:.4g}"
        y = rect.get_height() * 0.92
        va = "center"
        color = "black"
    ax2.text(
        rect.get_x() + rect.get_width() / 2,
        y,
        label,
        ha="center",
        va=va,
        fontsize=8,
        color=color,
    )

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "typeB_typeC_ratio_grouped.png"), dpi=200)
# plt.show()


#### Optional: flip-rate plotting helper (requires passing numbers) ####
def plot_flip_rate(model_label, flip_native, flip_romaji, savepath=None):
    """Plot flip rates (as percentages) for a single model comparing native vs romaji inputs.

    Example: plot_flip_rate('mDeBERTa (native-trained)', 0.05, 0.18, savepath='...')
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["native_input", "romaji_input"]
    vals = [flip_native * 100, flip_romaji * 100]
    ax.bar(labels, vals, color=["#2a92d6", "#f0b24a"])
    ax.set_ylabel("Flip rate (%)")
    ax.set_title(f"{model_label}: Flip rate by input script")
    ax.set_ylim(0, max(30, max(vals) + 5))
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=200)
    return fig


# Usage (uncomment and set real values):
# fig = plot_flip_rate('mDeBERTa (native-trained)', 0.05, 0.18, savepath=os.path.join(out_dir, 'mdeberta_flip_rate.png'))


# -------------------------
# Combined 2x2 figure
# Top-left: Accuracy
# Top-right: Missed Toxicity Rate (FN)
# Bottom-left: Type B / Type C (grouped)
# Bottom-right: Over-flagged Non-Toxic Rate (FP)
# -------------------------
figc, axesc = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Accuracy
ax_tl = axesc[0, 0]
bars_tl_a = ax_tl.bar(
    x - width / 2, acc_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_tl_b = ax_tl.bar(
    x + width / 2, acc_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
ax_tl.set_xticks(x)
ax_tl.set_xticklabels(models)
ax_tl.set_title("Accuracy on Romaji Test")
ax_tl.set_ylabel("Percent (%)")
ax_tl.legend()
annotate_bars([bars_tl_a, bars_tl_b], fmt="{:.2f}", offset=1.5)
ax_tl.grid(axis="y", linestyle="--", alpha=0.4)

# Top-right: Missed Toxicity (FN)
ax_tr = axesc[0, 1]
bars_tr_a = ax_tr.bar(
    x - width / 2, fn_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_tr_b = ax_tr.bar(
    x + width / 2, fn_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
ax_tr.set_xticks(x)
ax_tr.set_xticklabels(models)
ax_tr.set_title("Missed Toxicity (False Negatives %) on Romaji")
ax_tr.legend()
annotate_bars([bars_tr_a, bars_tr_b], fmt="{:.2f}", inside=True, inside_frac=0.18)
ax_tr.grid(axis="y", linestyle="--", alpha=0.4)

# Bottom-left: Type B / Type C  Ratio grouped
ax_bl = axesc[1, 0]
bars_bl_n = ax_bl.bar(
    x2 - width2 / 2, plot_native, width2, label="native-trained", color="#2a92d6"
)
bars_bl_r = ax_bl.bar(
    x2 + width2 / 2, plot_romaji, width2, label="romaji-trained", color="#f0b24a"
)
ax_bl.set_xticks(x2)
ax_bl.set_xticklabels(models_short)
# ax_bl.set_yscale("log")
ax_bl.set_ylabel("Type B / Type C Ratio")
ax_bl.set_title("Type B / Type C Ratio — Native vs Romaji-trained")
ax_bl.legend()
# Reference line at 1 for the grouped Type B/Type C plot
ax_bl.axhline(1.0, color="gray", linestyle="--", linewidth=1)
for r_val, rect in zip(ratios_native, bars_bl_n):
    if not np.isfinite(r_val):
        label = "∞"
        y = rect.get_height() * 0.98
        va = "center"
        color = "black"
    else:
        label = f"{r_val:.2f}"
        y = rect.get_height() * 0.92
        va = "center"
        color = "black"
    ax_bl.text(
        rect.get_x() + rect.get_width() / 2,
        y,
        label,
        ha="center",
        va=va,
        fontsize=9,
        color=color,
    )
for r_val, rect in zip(ratios_romaji, bars_bl_r):
    if not np.isfinite(r_val):
        label = "∞"
        y = rect.get_height() * 0.98
        va = "center"
        color = "black"
    else:
        label = f"{r_val:.4g}"
        y = rect.get_height() * 0.92
        va = "center"
        color = "black"
    ax_bl.text(
        rect.get_x() + rect.get_width() / 2,
        y,
        label,
        ha="center",
        va=va,
        fontsize=9,
        color=color,
    )

ax_bl.grid(axis="y", linestyle="--", alpha=0.4)

# Bottom-right: Over-Flagged Non-Toxic (FP)
ax_br = axesc[1, 1]
bars_br_a = ax_br.bar(
    x - width / 2, fp_native_tr, width, label="native-trained", color="#2a92d6"
)
bars_br_b = ax_br.bar(
    x + width / 2, fp_romaji_tr, width, label="romaji-trained", color="#f0b24a"
)
ax_br.set_xticks(x)
ax_br.set_xticklabels(models)
ax_br.set_title("Over-Flagged Non-Toxic (False Positives %) on Romaji")
ax_br.legend()

max_fp = (
    max(max(fp_native_tr), max(fp_romaji_tr)) if fp_native_tr and fp_romaji_tr else 1.0
)
# Use inside labels for FP and make a slightly tighter top margin
ax_br.set_ylim(0, max_fp * 1.15)
annotate_bars([bars_br_a, bars_br_b], fmt="{:.2f}", inside=True, inside_frac=0.18)
ax_br.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "combined_2x2.png"), dpi=200)
