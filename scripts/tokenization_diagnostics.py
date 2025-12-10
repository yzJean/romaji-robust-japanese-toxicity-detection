#!/usr/bin/env python3
"""
Tokenization diagnostics and flip-rate evaluation for tokenized models.

Usage examples:
  python3 scripts/tokenization_diagnostics.py \
    --csv data/processed/paired_native_romaji_inspection_ai_binary.csv \
    --model-name microsoft/mdeberta-v3-base \
    --checkpoint outputs/microsoft_mdeberta_v3_base_romaji_best_model.pt \
    --output diagnostics_mdeberta.json

The script computes:
  - Average tokens per sentence (native vs romaji)
  - Tokenization granularity: mean(tokens_romaji / tokens_native)
  - Unknown-token (OOV) counts if tokenizer provides unk token id
  - Model predictions on aligned native/romaji test set and flip rate
  - Contingency table for McNemar (n00, n01, n10, n11) and exact p-value if scipy available

"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

# Add src directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    SimpleBertClassifier,
    SimpleTrainer,
    SimpleToxicityDataset,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def safe_load_checkpoint(model, path, device):
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
        model.load_state_dict(state)
    elif isinstance(ck, dict):
        try:
            model.load_state_dict(ck)
        except Exception as e:
            raise RuntimeError(f"Unrecognized checkpoint format: {e}")
    else:
        raise RuntimeError("Unsupported checkpoint format")
    model.to(device)
    return ck


def tokenize_counts(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Tokenization diagnostics and flip-rate"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Paired CSV with text_native, text_romaji, label_int_coarse",
    )
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument(
        "--checkpoint",
        help="Optional path to a saved best_model .pt (contains model_state_dict)",
    )
    parser.add_argument(
        "--checkpoint-b",
        dest="checkpoint_b",
        help="Optional path to a second checkpoint to compare (evaluated on romaji)",
    )
    parser.add_argument(
        "--model-name-b",
        dest="model_name_b",
        help="Optional model name for the second checkpoint (defaults to --model-name)",
        default=None,
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="tokenization_diagnostics.json")

    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    df = pd.read_csv(args.csv)
    required = {"text_native", "text_romaji", "label_int_coarse"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    # Build a stratified test split on indices to preserve pairing
    labels = df["label_int_coarse"].tolist()
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=labels
    )
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Load tokenizer for the model under test
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tokenization diagnostics
    total_tokens_native = 0
    total_tokens_romaji = 0
    total_unk_native = 0
    total_unk_romaji = 0
    ratios = []
    per_sent_unk_native = []
    per_sent_unk_romaji = []

    # Determine unk token id if available
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_id is None and unk_token is not None:
        # Try to convert token to id
        try:
            unk_id = tokenizer.convert_tokens_to_ids(unk_token)
        except Exception:
            unk_id = None

    samples = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Tokenizing"):
        native = str(row["text_native"]) if pd.notna(row["text_native"]) else ""
        romaji = str(row["text_romaji"]) if pd.notna(row["text_romaji"]) else ""

        tokens_native = tokenize_counts(tokenizer, native)
        tokens_romaji = tokenize_counts(tokenizer, romaji)

        n_native = len(tokens_native)
        n_romaji = len(tokens_romaji)
        total_tokens_native += n_native
        total_tokens_romaji += n_romaji

        # Unknown token counts (if tokenizer uses unk_id)
        unk_n = 0
        unk_r = 0
        if unk_id is not None:
            ids_native = tokenizer.convert_tokens_to_ids(tokens_native)
            ids_romaji = tokenizer.convert_tokens_to_ids(tokens_romaji)
            unk_n = sum(1 for i in ids_native if i == unk_id)
            unk_r = sum(1 for i in ids_romaji if i == unk_id)

        total_unk_native += unk_n
        total_unk_romaji += unk_r
        per_sent_unk_native.append(unk_n)
        per_sent_unk_romaji.append(unk_r)

        # tokenization granularity ratio (avoid divide-by-zero)
        ratio = float(n_romaji) / max(1, n_native)
        ratios.append(ratio)

        samples.append(
            {
                "native": native,
                "romaji": romaji,
                "n_native": n_native,
                "n_romaji": n_romaji,
                "unk_native": unk_n,
                "unk_romaji": unk_r,
                "label": int(row["label_int_coarse"]),
            }
        )

    avg_tokens_native = total_tokens_native / max(1, len(test_df))
    avg_tokens_romaji = total_tokens_romaji / max(1, len(test_df))
    avg_ratio = float(np.mean(ratios))
    oov_rate_native = (
        total_unk_native / max(1, total_tokens_native)
        if total_tokens_native > 0
        else 0.0
    )
    oov_rate_romaji = (
        total_unk_romaji / max(1, total_tokens_romaji)
        if total_tokens_romaji > 0
        else 0.0
    )
    print("Tokenization Diagnostics:")
    print(
        f"  Total tokens native: {total_tokens_native:.2f}, total unk: {total_unk_native} ({oov_rate_native*100:.3f}%)"
    )
    print(
        f"  Total tokens romaji: {total_tokens_romaji:.2f}, total unk: {total_unk_romaji} ({oov_rate_romaji*100:.3f}%)"
    )

    # Prepare datasets and dataloaders for model inference
    test_texts_native = [s["native"] for s in samples]
    test_texts_romaji = [s["romaji"] for s in samples]
    test_labels = [s["label"] for s in samples]

    dataset_native = SimpleToxicityDataset(
        test_texts_native, test_labels, tokenizer, args.max_length
    )
    dataset_romaji = SimpleToxicityDataset(
        test_texts_romaji, test_labels, tokenizer, args.max_length
    )

    loader_native = DataLoader(
        dataset_native, batch_size=args.batch_size, shuffle=False
    )
    loader_romaji = DataLoader(
        dataset_romaji, batch_size=args.batch_size, shuffle=False
    )

    # Create model and optionally load checkpoint
    model = SimpleBertClassifier(args.model_name)
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        ck = safe_load_checkpoint(model, args.checkpoint, device)
    else:
        model.to(device)

    trainer = SimpleTrainer(model, device)

    # Run evaluation to get predictions
    _, acc_native, preds_native, labels_native = trainer.evaluate(loader_native)
    _, acc_romaji, preds_romaji, labels_romaji = trainer.evaluate(loader_romaji)

    preds_native = np.array(preds_native)
    preds_romaji = np.array(preds_romaji)
    labels_native = np.array(labels_native)

    # Flip rate
    if len(preds_native) != len(preds_romaji):
        raise RuntimeError(
            "Native and Romaji prediction lengths differ; alignment broken"
        )

    flip_mask = preds_native != preds_romaji
    flip_rate = float(flip_mask.sum()) / len(preds_native)

    # Directional flips counts (native->romaji)
    flip_pairs = Counter(zip(preds_native[flip_mask], preds_romaji[flip_mask]))
    # Convert tuple keys to strings for JSON serialization
    flip_counts_serializable = {f"{k[0]}->{k[1]}": v for k, v in flip_pairs.items()}

    # Contingency table for McNemar: n00 both correct, n01 native correct only, n10 romaji correct only, n11 both wrong
    native_correct = preds_native == labels_native
    romaji_correct = preds_romaji == labels_native

    n00 = int(np.sum(native_correct & romaji_correct))
    n01 = int(np.sum(native_correct & (~romaji_correct)))
    n10 = int(np.sum((~native_correct) & romaji_correct))
    n11 = int(np.sum((~native_correct) & (~romaji_correct)))
    print(
        f"Contingency Table:\n"
        f"                    Romaji Correct    Romaji Wrong\n"
        f"Native Correct          {n00}               {n01}\n"
        f"Native Wrong            {n10}               {n11}\n"
    )

    # McNemar exact p-value via binomial test if scipy available
    mcnemar_p = None
    b = n01
    n = n01 + n10

    if n > 0:
        # Try new scipy API (>= 1.7)
        from scipy.stats import binomtest

        result = binomtest(b, n, p=0.5, alternative="two-sided")
        mcnemar_p = float(result.pvalue)

        # Display results
        if mcnemar_p is not None:
            if mcnemar_p < 0.001:
                p_display = f"p < 0.001 (p = {mcnemar_p:.2e})"
            else:
                p_display = f"p = {mcnemar_p:.6f}"
            print(f"McNemar's test: n01={n01}, n10={n10}, {p_display}")
    else:
        print(f"McNemar's test skipped: no disagreements (n01={n01}, n10={n10})")
        mcnemar_p = None

    results = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "n_test": len(test_df),
        "avg_tokens_native": avg_tokens_native,
        "avg_tokens_romaji": avg_tokens_romaji,
        "avg_token_ratio_romaji_over_native": avg_ratio,
        "oov_rate_native": oov_rate_native,
        "oov_rate_romaji": oov_rate_romaji,
        "unknown_token_id": int(unk_id) if unk_id is not None else None,
        "accuracy_native": float(acc_native),
        "accuracy_romaji": float(acc_romaji),
        "flip_rate": float(flip_rate),
        "flip_counts": flip_counts_serializable,
        "contingency": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "mcnemar_p_value": mcnemar_p,
        "mcnemar_significant": mcnemar_p < 0.05 if mcnemar_p is not None else None,
        "mcnemar_interpretation": (
            "highly significant (p < 0.001)"
            if mcnemar_p is not None and mcnemar_p < 0.001
            else (
                "significant (p < 0.05)"
                if mcnemar_p is not None and mcnemar_p < 0.05
                else (
                    "not significant (p >= 0.05)"
                    if mcnemar_p is not None
                    else "not computed"
                )
            )
        ),
    }

    # Include raw predictions for downstream analysis
    try:
        results["preds_native"] = preds_native.tolist()
    except Exception:
        results["preds_native"] = None
    try:
        results["preds_romaji"] = preds_romaji.tolist()
    except Exception:
        results["preds_romaji"] = None

    # If a second checkpoint is provided, load it and evaluate on the romaji set
    if args.checkpoint_b:
        model_name_b = (
            args.model_name_b if args.model_name_b is not None else args.model_name
        )
        model_b = SimpleBertClassifier(model_name_b)
        print(f"Loading second checkpoint from {args.checkpoint_b}")
        safe_load_checkpoint(model_b, args.checkpoint_b, device)
        trainer_b = SimpleTrainer(model_b, device)
        _, acc_b_romaji, preds_b_romaji, labels_b = trainer_b.evaluate(loader_romaji)
        preds_b_romaji = np.array(preds_b_romaji)

        # Alignment check
        if len(preds_b_romaji) != len(preds_romaji):
            raise RuntimeError(
                "Checkpoint B produced different number of romaji predictions"
            )

        a_correct = preds_romaji == labels_native
        b_correct = preds_b_romaji == labels_native

        comp_n00 = int(np.sum(a_correct & b_correct))
        comp_n01 = int(np.sum(a_correct & (~b_correct)))
        comp_n10 = int(np.sum((~a_correct) & b_correct))
        comp_n11 = int(np.sum((~a_correct) & (~b_correct)))

        # Binomial test on disagreements where classifiers disagree about correctness
        comp_b = comp_n01
        comp_n = comp_n01 + comp_n10
        comp_p = None
        if comp_n > 0:
            from scipy.stats import binomtest

            comp_res = binomtest(comp_b, comp_n, p=0.5, alternative="two-sided")
            comp_p = float(comp_res.pvalue)

        results["compare_checkpoints_romaji"] = {
            "model_a_checkpoint": args.checkpoint,
            "model_b_checkpoint": args.checkpoint_b,
            "accuracy_model_a_romaji": float(acc_romaji),
            "accuracy_model_b_romaji": float(acc_b_romaji),
            "contingency": {
                "n00": comp_n00,
                "n01": comp_n01,
                "n10": comp_n10,
                "n11": comp_n11,
            },
            "mcnemar_p_value": comp_p,
            "mcnemar_significant": comp_p < 0.05 if comp_p is not None else None,
        }

    # Include prediction counts so downstream plotting can compute percentages
    try:
        results["n_preds_native"] = int(len(preds_native))
    except Exception:
        results["n_preds_native"] = None

    try:
        results["n_preds_romaji"] = int(len(preds_romaji))
    except Exception:
        results["n_preds_romaji"] = None

    # Optionally include top-k flip examples for manual inspection
    flip_indices = np.where(flip_mask)[0]
    top_examples = []
    for idx in flip_indices[:50]:
        top_examples.append(
            {
                "index": int(idx),
                "native": test_texts_native[idx],
                "romaji": test_texts_romaji[idx],
                "label": int(labels_native[idx]),
                "pred_native": int(preds_native[idx]),
                "pred_romaji": int(preds_romaji[idx]),
            }
        )
    results["top_flip_examples"] = top_examples

    # Write results
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Diagnostics saved to {args.output}")


if __name__ == "__main__":
    main()
