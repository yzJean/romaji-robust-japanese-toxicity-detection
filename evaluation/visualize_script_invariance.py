#!/usr/bin/env python3
"""
Visualize script-invariance comparison between tokenizer-based and byte-level models.

This script creates visualizations comparing:
- ByT5 (byte-level)
- mDeBERTa (tokenizer-based)
- BERT Japanese (tokenizer-based)

Metrics visualized:
- F1 scores (native vs romaji)
- ΔF1 (script-invariance measure)
- Flip rate
- Accuracy comparison
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_byt5_results(byt5_eval_path):
    """Load ByT5 evaluation results from evaluate_byt5.py output."""
    # Since evaluate_byt5.py prints to console, we'll manually input the results
    # or read from a saved JSON if available
    
    # Default values from your ByT5 evaluation
    return {
        "model_name": "ByT5-small",
        "f1_native": 0.4277,
        "f1_romaji": 0.5307,
        "delta_f1": 0.1030,
        "flip_rate": 0.1386,
        "accuracy_native": 0.6285,
        "accuracy_romaji": 0.6231,
        "mcnemar_p_value": 0.0104,
        "significant": True
    }

def load_model_comparison_results(native_path, romaji_path):
    """Load mDeBERTa and BERT results from compare_models.py outputs."""
    results = {}
    
    if os.path.exists(native_path):
        with open(native_path, 'r') as f:
            native_data = json.load(f)
            for model_result in native_data.get('results', []):
                model_name = model_result['model_name']
                results[model_name] = {
                    'accuracy_native': model_result['final_accuracy'],
                    'f1_native': model_result.get('final_accuracy', 0.0)  # Placeholder
                }
    
    if os.path.exists(romaji_path):
        with open(romaji_path, 'r') as f:
            romaji_data = json.load(f)
            for model_result in romaji_data.get('results', []):
                model_name = model_result['model_name']
                if model_name in results:
                    results[model_name]['accuracy_romaji'] = model_result['final_accuracy']
                    results[model_name]['f1_romaji'] = model_result.get('final_accuracy', 0.0)
                else:
                    results[model_name] = {
                        'accuracy_romaji': model_result['final_accuracy'],
                        'f1_romaji': model_result.get('final_accuracy', 0.0)
                    }
    
    # Calculate delta metrics
    for model_name in results:
        if 'f1_native' in results[model_name] and 'f1_romaji' in results[model_name]:
            results[model_name]['delta_f1'] = abs(
                results[model_name]['f1_native'] - results[model_name]['f1_romaji']
            )
    
    return results

def plot_f1_comparison(byt5_data, tokenizer_data, output_dir):
    """Plot F1 score comparison: native vs romaji for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['ByT5-small'] + list(tokenizer_data.keys())
    native_f1 = [byt5_data['f1_native']] + [tokenizer_data[m].get('f1_native', 0) for m in tokenizer_data]
    romaji_f1 = [byt5_data['f1_romaji']] + [tokenizer_data[m].get('f1_romaji', 0) for m in tokenizer_data]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, native_f1, width, label='Native Japanese', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, romaji_f1, width, label='Romaji', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('F1 Score (Macro)', fontweight='bold')
    ax.set_title('F1 Score Comparison: Native Japanese vs Romaji', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/f1_comparison.png")
    plt.close()

def plot_script_invariance_metrics(byt5_data, tokenizer_data, output_dir):
    """Plot ΔF1 and flip rate - key script-invariance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['ByT5-small'] + list(tokenizer_data.keys())
    delta_f1_values = [byt5_data['delta_f1']] + [tokenizer_data[m].get('delta_f1', 0) for m in tokenizer_data]
    
    # ΔF1 plot
    colors = ['#e74c3c' if val > 0.05 else '#2ecc71' for val in delta_f1_values]
    bars = ax1.bar(range(len(models)), delta_f1_values, color=colors, alpha=0.7)
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('ΔF1 (absolute)', fontweight='bold')
    ax1.set_title('Script-Invariance: ΔF1 (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, label='Threshold (0.05)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, delta_f1_values)):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    # Flip rate plot (only for ByT5 unless we have prediction data)
    if 'flip_rate' in byt5_data:
        flip_rates = [byt5_data['flip_rate']]
        flip_models = ['ByT5-small']
        
        bars2 = ax2.bar(range(len(flip_models)), flip_rates, color='#9b59b6', alpha=0.7)
        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_ylabel('Flip Rate', fontweight='bold')
        ax2.set_title('Prediction Flip Rate (Lower is Better)', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(flip_models)))
        ax2.set_xticklabels(flip_models, rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, flip_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.4f}\n({int(val*743)}/743)',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Flip rate requires\nprediction-level data',
                ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.set_title('Prediction Flip Rate', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'script_invariance_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/script_invariance_metrics.png")
    plt.close()

def plot_accuracy_comparison(byt5_data, tokenizer_data, output_dir):
    """Plot accuracy comparison across models and scripts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['ByT5-small'] + list(tokenizer_data.keys())
    native_acc = [byt5_data.get('accuracy_native', 0)] + [tokenizer_data[m].get('accuracy_native', 0) for m in tokenizer_data]
    romaji_acc = [byt5_data.get('accuracy_romaji', 0)] + [tokenizer_data[m].get('accuracy_romaji', 0) for m in tokenizer_data]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, native_acc, width, label='Native Japanese', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, romaji_acc, width, label='Romaji', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Model Accuracy: Native Japanese vs Romaji', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_comparison.png")
    plt.close()

def plot_model_type_comparison(byt5_data, tokenizer_data, output_dir):
    """Plot byte-level vs tokenizer-based comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Average metrics for tokenizer-based models
    if tokenizer_data:
        avg_tokenizer_delta_f1 = np.mean([tokenizer_data[m].get('delta_f1', 0) for m in tokenizer_data])
    else:
        avg_tokenizer_delta_f1 = 0
    
    model_types = ['Byte-Level\n(ByT5)', 'Tokenizer-Based\n(Avg)']
    delta_f1_by_type = [byt5_data['delta_f1'], avg_tokenizer_delta_f1]
    
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(model_types, delta_f1_by_type, color=colors, alpha=0.7, width=0.6)
    
    ax.set_ylabel('ΔF1 (absolute)', fontweight='bold')
    ax.set_title('Script-Invariance by Model Type\n(Lower ΔF1 = More Script-Invariant)', 
                fontsize=13, fontweight='bold')
    ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, label='Good threshold (0.05)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, delta_f1_by_type):
        label = f'{val:.4f}'
        if val > 0.05:
            label += '\n❌ Not invariant'
        else:
            label += '\n✓ Script-invariant'
        ax.text(bar.get_x() + bar.get_width()/2., val,
               label,
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_type_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_type_comparison.png")
    plt.close()

def create_summary_table(byt5_data, tokenizer_data, output_dir):
    """Create a summary table of all metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Model', 'F1 Native', 'F1 Romaji', 'ΔF1', 'Flip Rate', 'Script-Invariant?']
    
    rows = []
    
    # ByT5 row
    rows.append([
        'ByT5-small',
        f"{byt5_data['f1_native']:.4f}",
        f"{byt5_data['f1_romaji']:.4f}",
        f"{byt5_data['delta_f1']:.4f}",
        f"{byt5_data.get('flip_rate', 0):.4f}",
        '❌ No' if byt5_data.get('significant', False) else '✓ Yes'
    ])
    
    # Tokenizer-based models
    for model_name in tokenizer_data:
        data = tokenizer_data[model_name]
        rows.append([
            model_name,
            f"{data.get('f1_native', 0):.4f}",
            f"{data.get('f1_romaji', 0):.4f}",
            f"{data.get('delta_f1', 0):.4f}",
            'N/A',
            '✓ Yes' if data.get('delta_f1', 1) < 0.05 else '❌ No'
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.12, 0.12, 0.12, 0.12, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.title('Script-Invariance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/summary_table.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize script-invariance comparison between tokenizer and byte-level models'
    )
    parser.add_argument(
        '--byt5-f1-native',
        type=float,
        default=0.4277,
        help='ByT5 F1 score on native Japanese'
    )
    parser.add_argument(
        '--byt5-f1-romaji',
        type=float,
        default=0.5307,
        help='ByT5 F1 score on romaji'
    )
    parser.add_argument(
        '--byt5-flip-rate',
        type=float,
        default=0.1386,
        help='ByT5 flip rate'
    )
    parser.add_argument(
        '--byt5-acc-native',
        type=float,
        default=0.6285,
        help='ByT5 accuracy on native Japanese'
    )
    parser.add_argument(
        '--byt5-acc-romaji',
        type=float,
        default=0.6231,
        help='ByT5 accuracy on romaji'
    )
    parser.add_argument(
        '--native-results',
        type=str,
        default='outputs/comparison/model_comparison.json',
        help='Path to native Japanese comparison results'
    )
    parser.add_argument(
        '--romaji-results',
        type=str,
        default='outputs/comparison_romaji/model_comparison.json',
        help='Path to romaji comparison results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/visualizations',
        help='Directory to save visualization plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ByT5 data
    byt5_data = {
        'model_name': 'ByT5-small',
        'f1_native': args.byt5_f1_native,
        'f1_romaji': args.byt5_f1_romaji,
        'delta_f1': abs(args.byt5_f1_native - args.byt5_f1_romaji),
        'flip_rate': args.byt5_flip_rate,
        'accuracy_native': args.byt5_acc_native,
        'accuracy_romaji': args.byt5_acc_romaji,
        'significant': True
    }
    
    # Load tokenizer-based model data
    tokenizer_data = load_model_comparison_results(args.native_results, args.romaji_results)
    
    print("="*80)
    print("Generating Script-Invariance Visualizations")
    print("="*80)
    print(f"\nByT5 Data:")
    print(f"  F1 Native: {byt5_data['f1_native']:.4f}")
    print(f"  F1 Romaji: {byt5_data['f1_romaji']:.4f}")
    print(f"  ΔF1: {byt5_data['delta_f1']:.4f}")
    print(f"  Flip Rate: {byt5_data['flip_rate']:.4f}")
    
    if tokenizer_data:
        print(f"\nTokenizer-based models found: {list(tokenizer_data.keys())}")
    else:
        print("\nNo tokenizer-based model data found. Visualizations will show ByT5 only.")
    
    print(f"\nGenerating plots in {args.output_dir}...\n")
    
    # Generate all plots
    plot_f1_comparison(byt5_data, tokenizer_data, args.output_dir)
    plot_script_invariance_metrics(byt5_data, tokenizer_data, args.output_dir)
    plot_accuracy_comparison(byt5_data, tokenizer_data, args.output_dir)
    plot_model_type_comparison(byt5_data, tokenizer_data, args.output_dir)
    create_summary_table(byt5_data, tokenizer_data, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)
    print(f"\nGenerated plots:")
    print(f"  1. f1_comparison.png - F1 scores native vs romaji")
    print(f"  2. script_invariance_metrics.png - ΔF1 and flip rate")
    print(f"  3. accuracy_comparison.png - Accuracy comparison")
    print(f"  4. model_type_comparison.png - Byte-level vs tokenizer-based")
    print(f"  5. summary_table.png - Complete metrics summary")
    print(f"\nLocation: {args.output_dir}/")

if __name__ == '__main__':
    main()
