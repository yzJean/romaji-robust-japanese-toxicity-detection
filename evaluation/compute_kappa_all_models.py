"""
Compute Cohen's Kappa for script-invariance (Native vs Romaji) for all models.
Kappa measures agreement between native and romaji predictions.
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import json
import os

def compute_kappa_for_model(ground_truth, predictions, model_name):
    """
    Compute Cohen's Kappa between ground truth and model predictions.
    
    Args:
        ground_truth: Array of ground truth labels (0/1)
        predictions: Array of model predictions (0/1)
        model_name: Name of model
    
    Returns:
        dict with kappa and interpretation
    """
    
    kappa = cohen_kappa_score(ground_truth, predictions)
    
    # Interpretation
    if kappa >= 0.81:
        interpretation = "Almost Perfect Agreement (κ ≥ 0.81)"
        quality = "Excellent"
    elif kappa >= 0.61:
        interpretation = "Substantial Agreement (0.61 ≤ κ ≤ 0.80)"
        quality = "Good"
    elif kappa >= 0.41:
        interpretation = "Moderate Agreement (0.41 ≤ κ ≤ 0.60)"
        quality = "Fair"
    elif kappa >= 0.21:
        interpretation = "Fair Agreement (0.21 ≤ κ ≤ 0.40)"
        quality = "Poor"
    else:
        interpretation = "Slight/Negative Agreement (κ < 0.21)"
        quality = "Very Poor"
    
    # Count agreement
    agreement_count = (ground_truth == predictions).sum()
    total = len(ground_truth)
    
    return {
        'model': model_name,
        'kappa': float(kappa),
        'interpretation': interpretation,
        'quality': quality,
        'same_predictions': int(agreement_count),
        'different_predictions': int(total - agreement_count),
        'percent_agreement': float(agreement_count / total * 100),
        'total_samples': int(total)
    }

def main():
    print("=" * 80)
    print("COHEN'S KAPPA: SCRIPT-INVARIANCE ANALYSIS FOR ALL MODELS")
    print("=" * 80)
    
    all_kappa_results = {}
    
    # ========== ByT5 ==========
    print("\n" + "-" * 80)
    print("ByT5 (Byte-level Model)")
    print("-" * 80)
    
    try:
        byt5_df = pd.read_csv('outputs/byt5_cross_architecture_evaluation_standardized.csv')
        gt = byt5_df['ToxicityGT'].values
        pred = byt5_df['ResultsFromByT5'].values
        
        byt5_kappa = compute_kappa_for_model(gt, pred, 'ByT5-small')
        all_kappa_results['ByT5-small'] = byt5_kappa
        
        print(f"\nModel: {byt5_kappa['model']}")
        print(f"Cohen's Kappa: {byt5_kappa['kappa']:.4f}")
        print(f"Interpretation: {byt5_kappa['interpretation']}")
        print(f"Quality: {byt5_kappa['quality']}")
        print(f"Accuracy: {byt5_df['IsTruePositive'].sum() / len(byt5_df) * 100:.2f}%")
        print(f"\nAgreement Statistics:")
        print(f"  Same predictions: {byt5_kappa['same_predictions']}/{byt5_kappa['total_samples']} ({byt5_kappa['percent_agreement']:.2f}%)")
        print(f"  Different predictions: {byt5_kappa['different_predictions']}/{byt5_kappa['total_samples']} ({100-byt5_kappa['percent_agreement']:.2f}%)")
        
    except FileNotFoundError:
        print("⚠️  ByT5 file not found: outputs/byt5_cross_architecture_evaluation_standardized.csv")
    except Exception as e:
        print(f"❌ Error processing ByT5: {e}")
    
    # ========== BERT ==========
    print("\n" + "-" * 80)
    print("BERT-Japanese (Tokenizer-based Model)")
    print("-" * 80)
    
    try:
        bert_df = pd.read_csv('outputs/bert_cross_architecture_evaluation.csv')
        gt = bert_df['ToxicityGT'].values
        pred = bert_df['ResultsFromBERT'].values
        
        bert_kappa = compute_kappa_for_model(gt, pred, 'BERT-Japanese')
        all_kappa_results['BERT-Japanese'] = bert_kappa
        
        print(f"\nModel: {bert_kappa['model']}")
        print(f"Cohen's Kappa: {bert_kappa['kappa']:.4f}")
        print(f"Interpretation: {bert_kappa['interpretation']}")
        print(f"Quality: {bert_kappa['quality']}")
        print(f"Accuracy: {bert_df['IsTruePositive'].sum() / len(bert_df) * 100:.2f}%")
        print(f"\nAgreement Statistics:")
        print(f"  Same predictions: {bert_kappa['same_predictions']}/{bert_kappa['total_samples']} ({bert_kappa['percent_agreement']:.2f}%)")
        print(f"  Different predictions: {bert_kappa['different_predictions']}/{bert_kappa['total_samples']} ({100-bert_kappa['percent_agreement']:.2f}%)")
        
    except FileNotFoundError:
        print("⚠️  BERT file not found: outputs/bert_cross_architecture_evaluation.csv")
    except Exception as e:
        print(f"❌ Error processing BERT: {e}")
    
    # ========== mDeBERTa ==========
    print("\n" + "-" * 80)
    print("mDeBERTa-v3 (Tokenizer-based Model)")
    print("-" * 80)
    
    try:
        mdeberta_df = pd.read_csv('outputs/mdeberta_cross_architecture_evaluation.csv')
        gt = mdeberta_df['ToxicityGT'].values
        pred = mdeberta_df['ResultsFrommDeBERTa'].values
        
        mdeberta_kappa = compute_kappa_for_model(gt, pred, 'mDeBERTa-v3-small')
        all_kappa_results['mDeBERTa-v3-small'] = mdeberta_kappa
        
        print(f"\nModel: {mdeberta_kappa['model']}")
        print(f"Cohen's Kappa: {mdeberta_kappa['kappa']:.4f}")
        print(f"Interpretation: {mdeberta_kappa['interpretation']}")
        print(f"Quality: {mdeberta_kappa['quality']}")
        print(f"Accuracy: {mdeberta_df['IsTruePositive'].sum() / len(mdeberta_df) * 100:.2f}%")
        print(f"\nAgreement Statistics:")
        print(f"  Same predictions: {mdeberta_kappa['same_predictions']}/{mdeberta_kappa['total_samples']} ({mdeberta_kappa['percent_agreement']:.2f}%)")
        print(f"  Different predictions: {mdeberta_kappa['different_predictions']}/{mdeberta_kappa['total_samples']} ({100-mdeberta_kappa['percent_agreement']:.2f}%)")
        
    except FileNotFoundError:
        print("⚠️  mDeBERTa file not found: outputs/mdeberta_cross_architecture_evaluation.csv")
    except Exception as e:
        print(f"❌ Error processing mDeBERTa: {e}")
    
    # ========== SUMMARY COMPARISON ==========
    if len(all_kappa_results) > 0:
        print("\n" + "=" * 80)
        print("SUMMARY: KAPPA COMPARISON ACROSS ALL MODELS")
        print("=" * 80)
        
        print(f"\n{'Model':<25} {'Kappa':<10} {'Quality':<15} {'% Agreement':<12}")
        print("-" * 80)
        
        for model_name, result in all_kappa_results.items():
            print(f"{result['model']:<25} {result['kappa']:<10.4f} {result['quality']:<15} {result['percent_agreement']:<12.2f}%")
        
        # Ranking
        print("\n" + "-" * 80)
        print("RANKING (Best to Worst Script-Invariance):")
        print("-" * 80)
        
        sorted_models = sorted(all_kappa_results.items(), 
                             key=lambda x: x[1]['kappa'], 
                             reverse=True)
        
        for rank, (model_name, result) in enumerate(sorted_models, 1):
            print(f"{rank}. {result['model']:<25} κ = {result['kappa']:>7.4f} ({result['quality']})")
        
        # Key insights
        print("\n" + "-" * 80)
        print("KEY INSIGHTS:")
        print("-" * 80)
        
        best_model = sorted_models[0][1]
        worst_model = sorted_models[-1][1]
        
        print(f"\n✅ Best script-invariance: {best_model['model']} (κ = {best_model['kappa']:.4f})")
        print(f"   → {best_model['interpretation']}")
        
        print(f"\n❌ Worst script-invariance: {worst_model['model']} (κ = {worst_model['kappa']:.4f})")
        print(f"   → {worst_model['interpretation']}")
        
        # Save to JSON
        output_file = 'outputs/kappa_script_invariance_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_kappa_results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("KAPPA COMPUTATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
