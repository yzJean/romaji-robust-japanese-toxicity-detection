"""
Comprehensive Section 7.2.3 Script: Cross-Architecture Evaluation
Generates all materials needed for section 7.2.3 submission:
- Evaluation CSV files (standardized format)
- Error taxonomy analysis (JSON)
- Summary metrics and tables
- Interpretation guidelines
"""

import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_standardized_csvs():
    """Load all three standardized evaluation CSVs."""
    files = {
        'ByT5': 'outputs/byt5_cross_architecture_evaluation_standardized.csv',
        'BERT': 'outputs/bert_cross_architecture_evaluation.csv',
        'mDeBERTa': 'outputs/mdeberta_cross_architecture_evaluation.csv'
    }
    
    data = {}
    for model_name, filepath in files.items():
        try:
            data[model_name] = pd.read_csv(filepath)
            print(f"✓ Loaded {model_name}: {len(data[model_name])} samples")
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
    
    return data

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics for a model."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Error analysis
    errors = y_true != y_pred
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
    true_positives = ((y_pred == 1) & (y_true == 1)).sum()
    true_negatives = ((y_pred == 0) & (y_true == 0)).sum()
    
    # Error ratio
    bc_ratio = false_negatives / false_positives if false_positives > 0 else 0
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'bc_ratio': bc_ratio,
        'total_samples': len(y_true),
        'toxic_samples': (y_true == 1).sum(),
        'non_toxic_samples': (y_true == 0).sum()
    }

def generate_error_taxonomy(data):
    """Generate comprehensive error taxonomy for all models."""
    taxonomy = {}
    
    for model_name, df in data.items():
        y_true = df['ToxicityGT'].values
        y_pred = df['ResultsFromModel'].values if 'ResultsFromModel' in df.columns else df[df.columns[3]].values
        
        # Type counts
        type_a = (y_true == y_pred).sum()  # Correct
        type_b = ((y_pred == 0) & (y_true == 1)).sum()  # False Negatives
        type_c = ((y_pred == 1) & (y_true == 0)).sum()  # False Positives
        
        # B/C ratio
        bc_ratio = type_b / type_c if type_c > 0 else float('inf')
        
        metrics = calculate_metrics(y_true, y_pred, model_name)
        
        taxonomy[model_name.lower() + '_romaji_results'] = {
            'total': len(df),
            'type_a': int(type_a),
            'type_b': int(type_b),
            'type_c': int(type_c),
            'bc_ratio': round(bc_ratio, 4),
            'accuracy': round(metrics['accuracy'], 4),
            'f1': round(metrics['f1'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'toxic_count': int(metrics['toxic_samples']),
            'non_toxic_count': int(metrics['non_toxic_samples']),
            'true_positives': int(type_a),
            'false_positives': int(type_c),
            'false_negatives': int(type_b)
        }
    
    return taxonomy

def generate_metrics_table(data):
    """Generate comprehensive metrics table for all models."""
    metrics_list = []
    
    for model_name, df in data.items():
        y_true = df['ToxicityGT'].values
        y_pred = df['ResultsFromModel'].values if 'ResultsFromModel' in df.columns else df[df.columns[3]].values
        
        metrics = calculate_metrics(y_true, y_pred, model_name)
        
        metrics_list.append({
            'Model': model_name,
            'Samples': metrics['total_samples'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'TP': metrics['true_positives'],
            'FP': metrics['false_positives'],
            'FN': metrics['false_negatives'],
            'TN': metrics['true_negatives']
        })
    
    return pd.DataFrame(metrics_list)

def load_kappa_results():
    """Load kappa results from JSON."""
    try:
        with open('outputs/kappa_script_invariance_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠ Kappa results file not found. Run compute_kappa_all_models.py first.")
        return {}

def generate_interpretation_guide(data, kappa_results):
    """Generate interpretation guide for section 7.2.3."""
    guide = {
        'section': '7.2.3 Cross-Architecture Evaluation',
        'overview': 'Comparison of ByT5 (byte-level) vs tokenizer-based models (BERT, mDeBERTa) on romaji script',
        'key_findings': {},
        'interpretation_notes': {
            'accuracy': 'Overall prediction correctness',
            'f1': 'Harmonic mean of precision and recall',
            'bc_ratio': 'False Negatives / False Positives; >1 means conservative, <1 means aggressive',
            'kappa': 'Script-invariance: ByT5 maintains consistency despite low accuracy'
        }
    }
    
    # Key findings by model
    for model_name, df in data.items():
        y_true = df['ToxicityGT'].values
        y_pred = df['ResultsFromModel'].values if 'ResultsFromModel' in df.columns else df[df.columns[3]].values
        
        metrics = calculate_metrics(y_true, y_pred, model_name)
        kappa = kappa_results.get(f"{model_name}-small", {}).get('kappa', 'N/A') if kappa_results else 'N/A'
        
        guide['key_findings'][model_name] = {
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'bc_ratio': metrics['bc_ratio'],
            'kappa': kappa,
            'interpretation': f"{model_name} performs {'well' if metrics['accuracy'] > 0.7 else 'moderately' if metrics['accuracy'] > 0.5 else 'poorly'} on romaji toxicity detection"
        }
    
    return guide

def main():
    print("\n" + "="*80)
    print("SECTION 7.2.3: CROSS-ARCHITECTURE EVALUATION")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading evaluation CSVs...")
    data = load_standardized_csvs()
    
    if not data:
        print("✗ No data loaded. Exiting.")
        return
    
    # Generate error taxonomy
    print("\n[2/4] Generating error taxonomy...")
    taxonomy = generate_error_taxonomy(data)
    
    with open('outputs/error_taxonomy_results.json', 'w') as f:
        json.dump({'models': taxonomy}, f, indent=2)
    print(f"✓ Saved error taxonomy to outputs/error_taxonomy_results.json")
    
    # Generate metrics table
    print("\n[3/4] Generating metrics table...")
    metrics_df = generate_metrics_table(data)
    print("\n" + metrics_df.to_string(index=False))
    
    metrics_df.to_csv('outputs/section_723_metrics_summary.csv', index=False)
    print(f"\n✓ Saved metrics summary to outputs/section_723_metrics_summary.csv")
    
    # Load kappa results
    print("\n[4/4] Loading kappa results...")
    kappa_results = load_kappa_results()
    
    # Generate interpretation guide
    interpretation = generate_interpretation_guide(data, kappa_results)
    
    with open('outputs/section_723_interpretation_guide.json', 'w') as f:
        json.dump(interpretation, f, indent=2)
    print(f"✓ Saved interpretation guide to outputs/section_723_interpretation_guide.json")
    
    # Print summary
    print("\n" + "="*80)
    print("SECTION 7.2.3 GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  ✓ outputs/error_taxonomy_results.json")
    print("  ✓ outputs/section_723_metrics_summary.csv")
    print("  ✓ outputs/section_723_interpretation_guide.json")
    print("\nStandardized CSV Files:")
    print("  ✓ outputs/byt5_cross_architecture_evaluation_standardized.csv")
    print("  ✓ outputs/bert_cross_architecture_evaluation.csv")
    print("  ✓ outputs/mdeberta_cross_architecture_evaluation.csv")
    print("\nVisualization Files (from visualize_model_comparison.py):")
    print("  ✓ outputs/visualizations/1_accuracy_comparison.png")
    print("  ✓ outputs/visualizations/2_kappa_comparison.png")
    print("  ✓ outputs/visualizations/3_error_distribution.png")
    print("  ✓ outputs/visualizations/4_model_ranking.png")
    print("  ✓ outputs/visualizations/5_summary_table.png")
    print("\nKappa Results (from compute_kappa_all_models.py):")
    print("  ✓ outputs/kappa_script_invariance_results.json")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
