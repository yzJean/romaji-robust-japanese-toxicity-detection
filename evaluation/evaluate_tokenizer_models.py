#!/usr/bin/env python3
"""
Evaluate script-invariance for tokenizer-based models (mDeBERTa, BERT Japanese).

Similar to evaluate_byt5.py but for tokenizer-based models.
Computes F1 scores, ΔF1, flip rate, and McNemar's test.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import chi2
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_data, SimpleToxicityDataset

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model name from checkpoint
    model_name = checkpoint.get('tokenizer_name') or checkpoint.get('model_name')
    if not model_name:
        raise ValueError("Could not determine model name from checkpoint")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("No model state dict found in checkpoint")
    
    model.to(device)
    model.eval()
    
    return model, model_name

def get_predictions(model, tokenizer, texts, labels, max_length, batch_size, device):
    """Get model predictions on a dataset."""
    dataset = SimpleToxicityDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels_batch.cpu().numpy().tolist())
    
    return np.array(all_preds), np.array(all_labels)

def compute_script_invariance(native_preds, romaji_preds, labels):
    """Compute script-invariance metrics."""
    native_preds = np.array(native_preds)
    romaji_preds = np.array(romaji_preds)
    labels = np.array(labels)
    
    # F1 scores
    f1_native = f1_score(labels, native_preds, average='macro', zero_division=0)
    f1_romaji = f1_score(labels, romaji_preds, average='macro', zero_division=0)
    delta_f1 = abs(f1_native - f1_romaji)
    
    # Flip rate
    flips = (native_preds != romaji_preds).sum()
    flip_rate = flips / len(labels)
    
    # McNemar's test
    native_correct = (native_preds == labels)
    romaji_correct = (romaji_preds == labels)
    
    both_correct = (native_correct & romaji_correct).sum()
    native_only = (native_correct & ~romaji_correct).sum()
    romaji_only = (~native_correct & romaji_correct).sum()
    both_wrong = (~native_correct & ~romaji_correct).sum()
    
    # McNemar's test statistic
    if native_only + romaji_only > 0:
        mcnemar_stat = ((abs(native_only - romaji_only) - 1) ** 2) / (native_only + romaji_only)
        p_value = 1 - chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0.0
        p_value = 1.0
    
    return {
        "f1_native": float(f1_native),
        "f1_romaji": float(f1_romaji),
        "delta_f1": float(delta_f1),
        "flip_rate": float(flip_rate),
        "num_flips": int(flips),
        "total_samples": int(len(labels)),
        "mcnemar": {
            "statistic": float(mcnemar_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "contingency": {
                "both_correct": int(both_correct),
                "native_correct_romaji_wrong": int(native_only),
                "native_wrong_romaji_correct": int(romaji_only),
                "both_wrong": int(both_wrong),
            }
        }
    }

def print_results(model_name, metrics):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"{model_name} Script-Invariance Evaluation Results")
    print(f"{'='*80}")
    print(f"F1-score (Native):    {metrics['f1_native']:.4f}")
    print(f"F1-score (Romaji):    {metrics['f1_romaji']:.4f}")
    print(f"ΔF1 (absolute):       {metrics['delta_f1']:.4f}")
    print(f"Flip Rate:            {metrics['flip_rate']:.4f} ({metrics['num_flips']}/{metrics['total_samples']} samples)")
    print(f"\nMcNemar's Test (Statistical Significance):")
    print(f"  Statistic:          {metrics['mcnemar']['statistic']:.4f}")
    print(f"  p-value:            {metrics['mcnemar']['p_value']:.4f}")
    print(f"  Significant (α=0.05): {'Yes' if metrics['mcnemar']['significant'] else 'No'}")
    print(f"\nContingency Table:")
    print(f"  Both Correct:       {metrics['mcnemar']['contingency']['both_correct']}")
    print(f"  Native✓ Romaji✗:   {metrics['mcnemar']['contingency']['native_correct_romaji_wrong']}")
    print(f"  Native✗ Romaji✓:   {metrics['mcnemar']['contingency']['native_wrong_romaji_correct']}")
    print(f"  Both Wrong:         {metrics['mcnemar']['contingency']['both_wrong']}")
    print(f"{'='*80}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate script-invariance for tokenizer-based models')
    parser.add_argument('--model-name', type=str, required=True, 
                       choices=['mdeberta', 'bert-japanese'],
                       help='Model to evaluate (mdeberta or bert-japanese)')
    parser.add_argument('--data-path', type=str,
                       default='data/processed/paired_native_romaji_llmjp_binary.csv',
                       help='Path to paired data')
    parser.add_argument('--native-checkpoint', type=str, required=True,
                       help='Path to native model checkpoint')
    parser.add_argument('--romaji-checkpoint', type=str, required=True,
                       help='Path to romaji model checkpoint')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Max sequence length')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluations',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading test data...")
    _, test_texts_native, _, test_labels = load_data(
        args.data_path,
        use_romaji=False,
        test_size=0.2
    )
    
    _, test_texts_romaji, _, _ = load_data(
        args.data_path,
        use_romaji=True,
        test_size=0.2
    )
    
    print(f"Test samples: {len(test_labels)}")
    
    # Load models
    print(f"\nLoading native model from {args.native_checkpoint}...")
    native_model, model_name = load_model(args.native_checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading romaji model from {args.romaji_checkpoint}...")
    romaji_model, _ = load_model(args.romaji_checkpoint, device)
    
    # Get predictions
    print("\nGetting predictions from native model...")
    native_preds, labels = get_predictions(
        native_model, tokenizer, test_texts_native, 
        test_labels, args.max_length, args.batch_size, device
    )
    
    print("Getting predictions from romaji model...")
    romaji_preds, _ = get_predictions(
        romaji_model, tokenizer, test_texts_romaji,
        test_labels, args.max_length, args.batch_size, device
    )
    
    # Compute metrics
    print("\nComputing script-invariance metrics...")
    metrics = compute_script_invariance(native_preds, romaji_preds, labels)
    
    # Print results
    print_results(args.model_name, metrics)
    
    # Save results
    output_file = os.path.join(args.output_dir, f'{args.model_name}_script_invariance.json')
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
