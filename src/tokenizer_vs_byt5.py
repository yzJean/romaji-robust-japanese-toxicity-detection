#!/usr/bin/env python3
"""Train and compare a subword tokenizer model vs the byte-level ByT5 model with script-invariance evaluation.

This script trains models on both native Japanese and romanized text, then compares:
  1. A baseline subword tokenizer model (default mDeBERTa-v3).
  2. A tokenizer-free byte-level model (google/byt5-small).

For each model, trains on both native and romaji to compute script-invariance metrics.
Results (metrics + checkpoints) are stored under the configured output directory.
"""

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy.stats import chi2

from utils import load_data, SimpleToxicityDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "paired_native_romaji_inspection_ai_binary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tokenizer_vs_byt5"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelSpec:
    name: str
    label: str
    max_length: int


def build_dataloaders(
    train_texts,
    test_texts,
    train_labels,
    test_labels,
    tokenizer,
    max_length: int,
    batch_size: int,
):
    train_dataset = SimpleToxicityDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SimpleToxicityDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate(model, loader, device) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    observed_labels = sorted(set(all_labels) | set(all_preds))
    # Ensure at least one label exists to avoid downstream errors
    if not observed_labels:
        observed_labels = [0]

    label_names_map = {0: "Non-Toxic", 1: "Toxic"}
    if len(observed_labels) == 2:
        report_labels = [0, 1]
        target_names = [label_names_map[lbl] for lbl in report_labels]
    else:
        report_labels = observed_labels
        target_names = [label_names_map.get(lbl, f"Class {lbl}") for lbl in report_labels]

    report = classification_report(
        all_labels,
        all_preds,
        labels=report_labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=report_labels).tolist()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
    }


def compute_script_invariance(native_preds: List[int], romaji_preds: List[int], 
                               labels: List[int]) -> Dict:
    """Compute script-invariance metrics between native and romaji predictions."""
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
    
    contingency = [[both_correct, native_only],
                   [romaji_only, both_wrong]]
    
    # McNemar's test statistic
    if native_only + romaji_only > 0:
        mcnemar_stat = ((abs(native_only - romaji_only) - 1) ** 2) / (native_only + romaji_only)
        # Chi-square distribution with 1 df
        from scipy.stats import chi2
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


def export_byte_eval_artifacts(
    export_dir: Path,
    safe_name: str,
    spec: ModelSpec,
    args,
    data_path: Path,
    native_checkpoint: Path,
    romaji_checkpoint: Path,
):
    """Export ByT5 checkpoints/configs so evaluation/evaluate_byt5.py can consume them."""
    export_dir.mkdir(parents=True, exist_ok=True)
    native_target = export_dir / f"{safe_name}_best_model.pt"
    romaji_target = export_dir / f"{safe_name}_romaji_best_model.pt"
    shutil.copy2(native_checkpoint, native_target)
    shutil.copy2(romaji_checkpoint, romaji_target)

    common_config = {
        "model_name": spec.name,
        "max_length": spec.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "seed": args.seed,
        "data_path": str(data_path),
    }
    native_config_path = export_dir / "byt5_config.json"
    romaji_config_path = export_dir / "byt5_romaji_config.json"
    with open(native_config_path, "w", encoding="utf-8") as f:
        json.dump({**common_config, "use_romaji": False}, f, indent=2)
    with open(romaji_config_path, "w", encoding="utf-8") as f:
        json.dump({**common_config, "use_romaji": True}, f, indent=2)

    print(
        f"[byte] Exported evaluate_byt5 artifacts to {export_dir} "
        f"(checkpoints + config JSONs)",
    )


def train_and_compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    data_path = Path(args.data_path).expanduser()
    if not data_path.is_absolute():
        data_path = (SCRIPT_DIR / data_path).resolve()

    # Load both native and romaji data for script-invariance testing
    train_texts_native, test_texts_native, train_labels, test_labels = load_data(
        str(data_path),
        use_romaji=False,
        test_size=args.test_size,
    )
    
    train_texts_romaji, test_texts_romaji, _, _ = load_data(
        str(data_path),
        use_romaji=True,
        test_size=args.test_size,
    )

    if args.sample_size:
        rng = random.Random(args.seed)
        paired_train = list(zip(train_texts_native, train_texts_romaji, train_labels))
        rng.shuffle(paired_train)
        paired_train = paired_train[: args.sample_size]
        train_texts_native, train_texts_romaji, train_labels = zip(*paired_train) if paired_train else ([], [], [])

        test_cap = max(10, args.sample_size // 4)
        paired_test = list(zip(test_texts_native, test_texts_romaji, test_labels))
        rng.shuffle(paired_test)
        paired_test = paired_test[:test_cap]
        test_texts_native, test_texts_romaji, test_labels = zip(*paired_test) if paired_test else ([], [], [])

        train_texts_native = list(train_texts_native)
        train_texts_romaji = list(train_texts_romaji)
        train_labels = list(train_labels)
        test_texts_native = list(test_texts_native)
        test_texts_romaji = list(test_texts_romaji)
        test_labels = list(test_labels)

    selected_models = list(dict.fromkeys([m.lower() for m in args.models]))
    specs: List[ModelSpec] = []
    if "subword" in selected_models:
        specs.append(
            ModelSpec(
                name=args.subword_model,
                label="subword",
                max_length=args.max_length,
            )
        )
    if "byte" in selected_models:
        specs.append(
            ModelSpec(
                name=args.byte_model,
                label="byte",
                max_length=args.byte_max_length or args.max_length,
            )
        )

    if not specs:
        raise ValueError("No models selected. Use --models with 'subword' and/or 'byte'.")

    byte_eval_export_dir = None
    if args.byte_eval_export_dir:
        byte_eval_export_dir = Path(args.byte_eval_export_dir).expanduser().resolve()
        byte_eval_export_dir.mkdir(parents=True, exist_ok=True)

    comparison = []

    for spec in specs:
        print(f"\n{'='*80}")
        print(f"Training {spec.label} model: {spec.name}")
        print(f"{'='*80}")
        
        tokenizer = AutoTokenizer.from_pretrained(spec.name)
        
        # Train on NATIVE text
        print(f"\n[{spec.label}] Training on NATIVE Japanese text...")
        train_loader_native, test_loader_native = build_dataloaders(
            train_texts_native,
            test_texts_native,
            train_labels,
            test_labels,
            tokenizer,
            spec.max_length,
            args.batch_size,
        )

        model_native = AutoModelForSequenceClassification.from_pretrained(
            spec.name,
            num_labels=2,
        )
        model_native.to(device)
        optimizer_native = AdamW(model_native.parameters(), lr=args.learning_rate)

        history_native = []
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model_native, train_loader_native, optimizer_native, device)
            val_metrics = evaluate(model_native, test_loader_native, device)
            history_native.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )
            print(
                f"[{spec.label}-native] Epoch {epoch}/{args.epochs} "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
            )

        elapsed_native = time.time() - start_time
        final_metrics_native = evaluate(model_native, test_loader_native, device)

        safe_name = spec.name.replace("/", "_").replace("-", "_")
        checkpoint_path_native = output_dir / f"{safe_name}_{spec.label}_native_best.pt"
        torch.save(
            {
                "model_state_dict": model_native.state_dict(),
                "tokenizer_name": spec.name,
                "config": vars(args),
                "history": history_native,
                "elapsed_sec": elapsed_native,
            },
            checkpoint_path_native,
        )
        
        # Clear GPU memory
        del model_native
        del optimizer_native
        torch.cuda.empty_cache()
        
        # Train on ROMAJI text
        print(f"\n[{spec.label}] Training on ROMAJI text...")
        train_loader_romaji, test_loader_romaji = build_dataloaders(
            train_texts_romaji,
            test_texts_romaji,
            train_labels,
            test_labels,
            tokenizer,
            spec.max_length,
            args.batch_size,
        )

        model_romaji = AutoModelForSequenceClassification.from_pretrained(
            spec.name,
            num_labels=2,
        )
        model_romaji.to(device)
        optimizer_romaji = AdamW(model_romaji.parameters(), lr=args.learning_rate)

        history_romaji = []
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model_romaji, train_loader_romaji, optimizer_romaji, device)
            val_metrics = evaluate(model_romaji, test_loader_romaji, device)
            history_romaji.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )
            print(
                f"[{spec.label}-romaji] Epoch {epoch}/{args.epochs} "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
            )

        elapsed_romaji = time.time() - start_time
        final_metrics_romaji = evaluate(model_romaji, test_loader_romaji, device)

        checkpoint_path_romaji = output_dir / f"{safe_name}_{spec.label}_romaji_best.pt"
        torch.save(
            {
                "model_state_dict": model_romaji.state_dict(),
                "tokenizer_name": spec.name,
                "config": vars(args),
                "history": history_romaji,
                "elapsed_sec": elapsed_romaji,
            },
            checkpoint_path_romaji,
        )
        
        # Clear GPU memory before computing metrics
        del model_romaji
        del optimizer_romaji
        torch.cuda.empty_cache()

        if spec.label == "byte" and byte_eval_export_dir:
            export_byte_eval_artifacts(
                byte_eval_export_dir,
                safe_name,
                spec,
                args,
                data_path,
                checkpoint_path_native,
                checkpoint_path_romaji,
            )
        
        # Compute script-invariance metrics
        print(f"\n[{spec.label}] Computing script-invariance metrics...")
        script_invariance = compute_script_invariance(
            final_metrics_native["predictions"],
            final_metrics_romaji["predictions"],
            test_labels
        )
        
        print(f"\n{spec.label.upper()} Script-Invariance Results:")
        print(f"  F1-score (Native): {script_invariance['f1_native']:.4f}")
        print(f"  F1-score (Romaji): {script_invariance['f1_romaji']:.4f}")
        print(f"  ΔF1 (absolute):    {script_invariance['delta_f1']:.4f}")
        print(f"  Flip Rate:         {script_invariance['flip_rate']:.4f} ({script_invariance['num_flips']}/{script_invariance['total_samples']} samples)")
        print(f"  McNemar p-value:   {script_invariance['mcnemar']['p_value']:.4f}")
        print(f"  Significant:       {'Yes' if script_invariance['mcnemar']['significant'] else 'No'}")

        comparison.append(
            {
                "label": spec.label,
                "model_name": spec.name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": spec.max_length,
                "native": {
                    "training_time_sec": elapsed_native,
                    "history": history_native,
                    "test_metrics": final_metrics_native,
                    "checkpoint_path": str(checkpoint_path_native),
                },
                "romaji": {
                    "training_time_sec": elapsed_romaji,
                    "history": history_romaji,
                    "test_metrics": final_metrics_romaji,
                    "checkpoint_path": str(checkpoint_path_romaji),
                },
                "script_invariance": script_invariance,
            }
        )

    summary_path = output_dir / "tokenizer_vs_byt5_script_invariance.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "comparison": comparison}, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("FINAL COMPARISON - Script-Invariance Metrics")
    print("="*80)
    for entry in comparison:
        print(f"\n{entry['label'].upper()} Model ({entry['model_name']}):")
        print(f"  Native Accuracy:  {entry['native']['test_metrics']['accuracy']:.4f}")
        print(f"  Romaji Accuracy:  {entry['romaji']['test_metrics']['accuracy']:.4f}")
        print(f"  ΔF1:              {entry['script_invariance']['delta_f1']:.4f}")
        print(f"  Flip Rate:        {entry['script_invariance']['flip_rate']:.4f}")
        print(f"  McNemar p-value:  {entry['script_invariance']['mcnemar']['p_value']:.4f}")
        print(f"  Script-Invariant: {'✓ YES' if not entry['script_invariance']['mcnemar']['significant'] else '✗ NO'}")
    
    print(f"\nResults saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tokenizer-based vs byte-level models for toxicity classification with script-invariance evaluation",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the processed CSV.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for AdamW.")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for the subword model.")
    parser.add_argument(
        "--byte-max-length",
        type=int,
        help="Optional override for ByT5 max sequence length (defaults to --max-length).",
    )
    parser.add_argument(
        "--subword-model",
        type=str,
        default="microsoft/mdeberta-v3-base",
        help="Tokenizer-based model identifier.",
    )
    parser.add_argument(
        "--byte-model",
        type=str,
        default="google/byt5-small",
        help="Tokenizer-free (byte-level) model identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store checkpoints and results.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional cap on training samples (test is scaled automatically).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["subword", "byte"],
        default=["subword", "byte"],
        help="Select which model families to train (default: both).",
    )
    parser.add_argument(
        "--byte-eval-export-dir",
        type=str,
        help="Optional directory to store evaluate_byt5.py-compatible ByT5 checkpoints/configs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_compare(args)
