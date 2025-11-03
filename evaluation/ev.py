import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "mock_eval_results.csv")
PLOT_PATH = os.path.join(BASE_DIR, "mock_f1_comparison.png")

# mock data for testing
np.random.seed(42)
N = 100
labels_true = np.random.randint(0, 2, N)
pred_native = np.random.randint(0, 2, N)
pred_romaji = np.random.randint(0, 2, N)


def compute_f1_scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_delta_f1(f1_native, f1_romaji):
    return f1_native - f1_romaji

def compute_flip_rate(pred_native, pred_romaji):
    same = np.sum(pred_native == pred_romaji)
    total = len(pred_native)
    return 1 - (same / total)

# computation
metrics_native = compute_f1_scores(labels_true, pred_native)
metrics_romaji = compute_f1_scores(labels_true, pred_romaji)
delta_f1 = compute_delta_f1(metrics_native["f1"], metrics_romaji["f1"])
flip_rate = compute_flip_rate(pred_native, pred_romaji)

summary = {
    "F1_native": metrics_native["f1"],
    "F1_romaji": metrics_romaji["f1"],
    "ΔF1": delta_f1,
    "FlipRate": flip_rate
}


df = pd.DataFrame([summary])
df.to_csv(CSV_PATH, index=False)
print(df.round(3))

labels = ['Native', 'Romaji']
values = [summary['F1_native'], summary['F1_romaji']]

plt.bar(labels, values, color=['skyblue', 'lightcoral'])
plt.ylabel('F1 Score')
plt.title('Mock F1 Comparison: Native vs Romaji')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()
