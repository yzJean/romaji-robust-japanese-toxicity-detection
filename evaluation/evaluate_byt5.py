"""
Evaluate ByT5 model on native vs romaji text.
Calculates F1-native, F1-romaji, ΔF1, and flip rate.
"""

import os
import sys
import json
import torch
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from scipy import stats
from tqdm import tqdm

# Use CuPy if CUDA is available, otherwise fall back to NumPy
try:
    import cupy as cp
    if torch.cuda.is_available():
        xp = cp
        print("Using CuPy for GPU-accelerated array operations")
    else:
        import numpy as np
        xp = np
        print("CUDA not available, using NumPy")
except ImportError:
    import numpy as np
    xp = np
    print("CuPy not installed, using NumPy")

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import SimpleBertClassifier, SimpleToxicityDataset, load_data

def load_model(model_path, config_path, device):
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if this is a ByT5 model - use float32 to avoid numerical issues
    is_byt5 = 'byt5' in config['model_name'].lower()
    
    model = SimpleBertClassifier(
        model_name=config['model_name'],
        dropout=config.get('dropout', 0.1),
        use_float32=is_byt5  # Force float32 for ByT5 models
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def get_predictions(model, dataloader, device):
    """Get model predictions on dataloader."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to appropriate array type (CuPy or NumPy)
    if xp.__name__ == 'cupy':
        return xp.array(all_preds), xp.array(all_labels)
    else:
        return xp.array(all_preds), xp.array(all_labels)

def compute_flip_rate(pred_native, pred_romaji):
    """Calculate percentage of predictions that flip between native and romaji."""
    flips = xp.sum(pred_native != pred_romaji)
    total = len(pred_native)
    # Convert to Python scalar if using CuPy
    if xp.__name__ == 'cupy':
        flips = float(flips.get())
    else:
        flips = float(flips)
    return flips / total

def mcnemar_test(y_true, pred_native, pred_romaji):
    """
    Perform McNemar's test to assess statistical significance of error rate differences.
    
    Returns:
        dict: Contains contingency table, test statistic, and p-value
    """
    # Create boolean arrays for correct predictions
    correct_native = (pred_native == y_true)
    correct_romaji = (pred_romaji == y_true)
    
    # Build 2x2 contingency table
    # a: both correct
    # b: native correct, romaji wrong
    # c: native wrong, romaji correct  
    # d: both wrong
    a = xp.sum(correct_native & correct_romaji)
    b = xp.sum(correct_native & ~correct_romaji)
    c = xp.sum(~correct_native & correct_romaji)
    d = xp.sum(~correct_native & ~correct_romaji)
    
    # Convert to Python scalars if using CuPy
    if xp.__name__ == 'cupy':
        a, b, c, d = int(a.get()), int(b.get()), int(c.get()), int(d.get())
    else:
        a, b, c, d = int(a), int(b), int(c), int(d)
    
    contingency_table = [[a, b], [c, d]]
    
    # McNemar's test statistic with continuity correction
    # Focus on discordant pairs (b and c)
    if b + c == 0:
        # No discordant pairs - models perform identically
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        "contingency_table": contingency_table,
        "both_correct": a,
        "native_correct_romaji_wrong": b,
        "native_wrong_romaji_correct": c,
        "both_wrong": d,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    data_path = os.path.join(base_dir, '..', 'data', 'processed', 
                             'paired_native_romaji_llmjp_binary.csv')
    
    # Model paths
    native_model_path = os.path.join(outputs_dir, 'google_byt5_small_best_model.pt')
    native_config_path = os.path.join(outputs_dir, 'byt5_config.json')
    romaji_model_path = os.path.join(outputs_dir, 'google_byt5_small_romaji_best_model.pt')
    romaji_config_path = os.path.join(outputs_dir, 'byt5_romaji_config.json')
    
    # Check if models exist
    if not os.path.exists(native_model_path):
        print(f"Error: Native model not found at {native_model_path}")
        return
    if not os.path.exists(romaji_model_path):
        print(f"Error: Romaji model not found at {romaji_model_path}")
        return
    
    # Load data
    print("\nLoading test data...")
    from transformers import AutoTokenizer
    
    # Load native model and tokenizer
    print("\nLoading native model...")
    native_model, native_config = load_model(native_model_path, native_config_path, device)
    tokenizer = AutoTokenizer.from_pretrained(native_config['model_name'])
    
    # Load test data (native text)
    _, test_texts_native, _, test_labels = load_data(
        data_path,
        use_romaji=False,
        test_size=0.2
    )
    
    # Load test data (romaji text)
    _, test_texts_romaji, _, _ = load_data(
        data_path,
        use_romaji=True,
        test_size=0.2
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    test_dataset_native = SimpleToxicityDataset(
        test_texts_native, test_labels, tokenizer, max_length=512
    )
    test_dataset_romaji = SimpleToxicityDataset(
        test_texts_romaji, test_labels, tokenizer, max_length=512
    )
    
    test_loader_native = DataLoader(test_dataset_native, batch_size=16, shuffle=False)
    test_loader_romaji = DataLoader(test_dataset_romaji, batch_size=16, shuffle=False)
    
    # Get predictions on native text
    print("\nEvaluating on native text...")
    pred_native, labels = get_predictions(native_model, test_loader_native, device)
    
    # Convert to NumPy for sklearn compatibility
    if xp.__name__ == 'cupy':
        labels_np = labels.get()
        pred_native_np = pred_native.get()
    else:
        labels_np = labels
        pred_native_np = pred_native
    
    f1_native = f1_score(labels_np, pred_native_np, average='macro')
    
    print("\nNative Text Results:")
    print(classification_report(labels_np, pred_native_np, 
                               target_names=['Non-Toxic', 'Toxic']))
    
    # Load romaji model
    print("\nLoading romaji model...")
    romaji_model, romaji_config = load_model(romaji_model_path, romaji_config_path, device)
    
    # Get predictions on romaji text
    print("\nEvaluating on romaji text...")
    pred_romaji, _ = get_predictions(romaji_model, test_loader_romaji, device)
    
    # Convert to NumPy for sklearn compatibility
    if xp.__name__ == 'cupy':
        pred_romaji_np = pred_romaji.get()
    else:
        pred_romaji_np = pred_romaji
    
    f1_romaji = f1_score(labels_np, pred_romaji_np, average='macro')
    
    print("\nRomaji Text Results:")
    print(classification_report(labels_np, pred_romaji_np, 
                               target_names=['Non-Toxic', 'Toxic']))
    
    # Calculate metrics
    delta_f1 = abs(f1_native - f1_romaji)
    flip_rate = compute_flip_rate(pred_native, pred_romaji)
    mcnemar_results = mcnemar_test(labels, pred_native, pred_romaji)
    
    # Calculate number of flips
    num_flips = xp.sum(pred_native != pred_romaji)
    if xp.__name__ == 'cupy':
        num_flips = int(num_flips.get())
        num_samples = int(len(labels))
    else:
        num_flips = int(num_flips)
        num_samples = len(labels)
    
    # Summary
    results = {
        "model": "ByT5-small",
        "F1_native": float(f1_native),
        "F1_romaji": float(f1_romaji),
        "ΔF1": float(delta_f1),
        "flip_rate": float(flip_rate),
        "num_samples": num_samples,
        "num_flips": num_flips,
        "mcnemar_test": mcnemar_results
    }
    
    print("\n" + "="*60)
    print("ByT5 Script-Invariance Evaluation Results")
    print("="*60)
    print(f"F1-score (Native):     {f1_native:.4f}")
    print(f"F1-score (Romaji):     {f1_romaji:.4f}")
    print(f"ΔF1 (absolute):        {delta_f1:.4f}")
    print(f"Flip Rate:             {flip_rate:.4f} ({results['num_flips']}/{results['num_samples']} samples)")
    print("\nMcNemar's Test (Statistical Significance):")
    print(f"  Statistic:           {mcnemar_results['statistic']:.4f}")
    print(f"  p-value:             {mcnemar_results['p_value']:.4f}")
    print(f"  Significant (α=0.05): {'Yes' if mcnemar_results['significant'] else 'No'}")
    print(f"\nContingency Table:")
    print(f"  Both Correct:        {mcnemar_results['both_correct']}")
    print(f"  Native✓ Romaji✗:     {mcnemar_results['native_correct_romaji_wrong']}")
    print(f"  Native✗ Romaji✓:     {mcnemar_results['native_wrong_romaji_correct']}")
    print(f"  Both Wrong:          {mcnemar_results['both_wrong']}")
    print("="*60)
    
    # Save results
    output_path = os.path.join(outputs_dir, 'byt5_script_invariance_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
