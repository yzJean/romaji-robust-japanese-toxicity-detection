#!/usr/bin/env python3
"""Train and compare a subword tokenizer model vs the byte-level ByT5 model.

This script reuses the processed binary toxicity data pipeline and trains two
models sequentially:
  1. A baseline subword tokenizer model (default mDeBERTa-v3).
  2. A tokenizer-free byte-level model (google/byt5-small).

Both models share the same train/test split to enable an apples-to-apples
comparison. Results (metrics + checkpoints) are stored under the configured
output directory.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    }


def train_and_compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    data_path = Path(args.data_path).expanduser()
    if not data_path.is_absolute():
        data_path = (SCRIPT_DIR / data_path).resolve()

    train_texts, test_texts, train_labels, test_labels = load_data(
        str(data_path),
        use_romaji=args.use_romaji,
        test_size=args.test_size,
    )

    if args.sample_size:
        rng = random.Random(args.seed)
        paired_train = list(zip(train_texts, train_labels))
        rng.shuffle(paired_train)
        paired_train = paired_train[: args.sample_size]
        train_texts, train_labels = zip(*paired_train) if paired_train else ([], [])

        test_cap = max(10, args.sample_size // 4)
        paired_test = list(zip(test_texts, test_labels))
        rng.shuffle(paired_test)
        paired_test = paired_test[:test_cap]
        test_texts, test_labels = zip(*paired_test) if paired_test else ([], [])

        train_texts = list(train_texts)
        train_labels = list(train_labels)
        test_texts = list(test_texts)
        test_labels = list(test_labels)

    specs = [
        ModelSpec(
            name=args.subword_model,
            label="subword",
            max_length=args.max_length,
        ),
        ModelSpec(
            name=args.byte_model,
            label="byte",
            max_length=args.byte_max_length or args.max_length,
        ),
    ]

    comparison = []

    for spec in specs:
        tokenizer = AutoTokenizer.from_pretrained(spec.name)
        train_loader, test_loader = build_dataloaders(
            train_texts,
            test_texts,
            train_labels,
            test_labels,
            tokenizer,
            spec.max_length,
            args.batch_size,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            spec.name,
            num_labels=2,
        )
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        history = []
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, test_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )
            print(
                f"[{spec.label}] Epoch {epoch}/{args.epochs} "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}"
            )

        elapsed = time.time() - start_time
        final_metrics = evaluate(model, test_loader, device)

        safe_name = spec.name.replace("/", "_").replace("-", "_")
        checkpoint_path = output_dir / f"{safe_name}_{spec.label}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer_name": spec.name,
                "config": vars(args),
                "history": history,
                "elapsed_sec": elapsed,
            },
            checkpoint_path,
        )

        comparison.append(
            {
                "label": spec.label,
                "model_name": spec.name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": spec.max_length,
                "training_time_sec": elapsed,
                "history": history,
                "test_metrics": final_metrics,
                "checkpoint_path": str(checkpoint_path),
            }
        )

    summary_path = output_dir / "tokenizer_vs_byt5_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "comparison": comparison}, f, indent=2, ensure_ascii=False)

    print("\nComparison complete. Results saved to:")
    print(f"  {summary_path}")
    for entry in comparison:
        print(
            f"  {entry['label']}: Acc={entry['test_metrics']['accuracy']:.4f} "
            f"Checkpoint={entry['checkpoint_path']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tokenizer-based vs byte-level models for toxicity classification",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the processed CSV.",
    )
    parser.add_argument(
        "--use-romaji",
        action="store_true",
        help="Use the romanized text column instead of native Japanese.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_compare(args)
