"""
Visualize Error Taxonomy: Create charts for slide presentation
Generates stacked bar charts and B/C ratio comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
models = ['ByT5', 'BERT', 'mDeBERTa']
type_a = [59.3, 78.5, 83.2]  # Correct
type_b = [16.4, 14.1, 10.2]  # False Negatives
type_c = [24.3, 7.4, 6.6]    # False Positives
bc_ratios = [0.68, 1.91, 1.55]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== SUBPLOT 1: Stacked Bar Chart =====
x = np.arange(len(models))
width = 0.5

bars1 = ax1.bar(x, type_a, width, label='Type A (Correct)', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, type_b, width, bottom=type_a, label='Type B (False Neg)', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x, type_c, width, bottom=np.array(type_a)+np.array(type_b), label='Type C (False Pos)', color='#f39c12', alpha=0.8)

ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Error Distribution by Model (Romaji)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add percentage labels on bars
for i, model in enumerate(models):
    ax1.text(i, type_a[i]/2, f'{type_a[i]:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax1.text(i, type_a[i] + type_b[i]/2, f'{type_b[i]:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)
    ax1.text(i, type_a[i] + type_b[i] + type_c[i]/2, f'{type_c[i]:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)

# ===== SUBPLOT 2: B/C Ratio Comparison =====
colors_bc = ['#e74c3c' if ratio < 1 else '#3498db' for ratio in bc_ratios]
bars_bc = ax2.barh(models, bc_ratios, color=colors_bc, alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('B/C Ratio', fontsize=12, fontweight='bold')
ax2.set_title('B/C Ratio: Model Behavior\n(< 1 = Aggressive | > 1 = Conservative)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 2.2)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Balanced (B/C=1)')

# Add value labels
for i, (model, ratio) in enumerate(zip(models, bc_ratios)):
    behavior = 'Aggressive' if ratio < 1 else 'Conservative' if ratio > 1.2 else 'Balanced'
    ax2.text(ratio + 0.05, i, f'{ratio:.2f}\n({behavior})', va='center', fontweight='bold', fontsize=10)

ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('outputs/error_taxonomy_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/error_taxonomy_visualization.png")

# Create a second figure: Detailed breakdown
fig2, ax3 = plt.subplots(figsize=(12, 6))

# Data for detailed breakdown
data = {
    'ByT5': {'Type A': 440, 'Type B': 122, 'Type C': 180},
    'BERT': {'Type A': 583, 'Type B': 105, 'Type C': 55},
    'mDeBERTa': {'Type A': 618, 'Type B': 76, 'Type C': 49}
}

x_pos = np.arange(len(models))
width = 0.25

type_a_counts = [440, 583, 618]
type_b_counts = [122, 105, 76]
type_c_counts = [180, 55, 49]

bars1 = ax3.bar(x_pos - width, type_a_counts, width, label='Type A (Correct)', color='#2ecc71', alpha=0.8)
bars2 = ax3.bar(x_pos, type_b_counts, width, label='Type B (False Neg)', color='#e74c3c', alpha=0.8)
bars3 = ax3.bar(x_pos + width, type_c_counts, width, label='Type C (False Pos)', color='#f39c12', alpha=0.8)

ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('Error Taxonomy: Absolute Counts (Romaji)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models, fontsize=11)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/error_taxonomy_counts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/error_taxonomy_counts.png")

# Summary statistics
print("\n" + "="*70)
print("ERROR TAXONOMY SUMMARY")
print("="*70)
print(f"\n{'Model':<12} {'Correct':<12} {'Missed':<12} {'Over-flagged':<15} {'B/C Ratio':<12} {'Behavior':<15}")
print("-"*70)
for i, model in enumerate(models):
    behavior = "Aggressive" if bc_ratios[i] < 1 else "Conservative" if bc_ratios[i] > 1.2 else "Balanced"
    print(f"{model:<12} {type_a[i]:>5.1f}% {type_b[i]:>10.1f}% {type_c[i]:>13.1f}% {bc_ratios[i]:>10.2f} {behavior:<15}")

print("\n" + "="*70)
print("✓ Visualization complete!")
print("="*70)
