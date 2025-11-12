#!/usr/bin/env python3
"""
Model comparison script for mDeBERTa-v3 vs BERT Japanese on Japanese toxicity classification.

Usage:
    python3 compare_models.py --quick-test           # Quick comparison with small data
    python3 compare_models.py --sample-size 200     # Compare with specific sample size
    python3 compare_models.py                        # Full comparison
"""

import argparse
import torch
import logging
import os
import time
from utils import load_data, SimpleToxicityDataset, SimpleBertClassifier, SimpleTrainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS = {
    "mDeBERTa-v3": "microsoft/mdeberta-v3-base",
    "BERT-Japanese": "tohoku-nlp/bert-base-japanese-v3",
}


def train_and_evaluate_model(
    model_name, model_display_name, train_loader, test_loader, device, args
):
    """Train and evaluate a single model."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_display_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Create model
    model = SimpleBertClassifier(model_name, dropout=args.dropout)

    # Create trainer
    trainer = SimpleTrainer(model, device, args.learning_rate)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc, predictions, true_labels = trainer.evaluate(test_loader)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_predictions = predictions
            best_true_labels = true_labels

    training_time = time.time() - start_time

    # Calculate final metrics
    final_accuracy = accuracy_score(best_true_labels, best_predictions)

    logger.info(f"\n{model_display_name} Results:")
    logger.info(f"Training time: {training_time:.1f}s")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final accuracy: {final_accuracy:.4f}")

    return {
        "model_name": model_display_name,
        "model_path": model_name,
        "training_time": training_time,
        "best_val_accuracy": best_val_acc,
        "final_accuracy": final_accuracy,
        "predictions": best_predictions,
        "true_labels": best_true_labels,
    }


def compare_models(args):
    """Compare BERT and XLM-RoBERTa models."""

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Quick test mode adjustments
    if args.quick_test:
        args.sample_size = 100  # Larger for comparison
        args.epochs = 2
        args.batch_size = 8
        logger.info("Quick comparison mode: 100 samples, 2 epochs")

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    train_texts, test_texts, train_labels, test_labels = load_data(
        args.data_path, use_romaji=args.use_romaji, test_size=args.test_size
    )

    # Limit data size if requested
    if args.sample_size:
        original_size = len(train_texts)
        train_texts = train_texts[: args.sample_size]
        train_labels = train_labels[: args.sample_size]

        test_sample_size = max(20, args.sample_size // 4)
        test_texts = test_texts[:test_sample_size]
        test_labels = test_labels[:test_sample_size]

        logger.info(
            f"Using {len(train_texts)} train samples, {len(test_texts)} test samples"
        )

    results = []

    # Train and evaluate each model
    for model_display_name, model_name in MODELS.items():
        try:
            logger.info(f"Loading tokenizer for {model_display_name}: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Create datasets
            train_dataset = SimpleToxicityDataset(
                train_texts, train_labels, tokenizer, args.max_length
            )
            test_dataset = SimpleToxicityDataset(
                test_texts, test_labels, tokenizer, args.max_length
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False
            )

            # Train and evaluate
            result = train_and_evaluate_model(
                model_name, model_display_name, train_loader, test_loader, device, args
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Error with {model_display_name}: {e}")
            continue

    # Print comparison results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    if len(results) >= 2:
        df_results = pd.DataFrame(
            [
                {
                    "Model": r["model_name"],
                    "Training Time (s)": f"{r['training_time']:.1f}",
                    "Best Val Accuracy": f"{r['best_val_accuracy']:.4f}",
                    "Final Accuracy": f"{r['final_accuracy']:.4f}",
                }
                for r in results
            ]
        )

        print(df_results.to_string(index=False))

        # Find best model
        best_model = max(results, key=lambda x: x["final_accuracy"])
        print(f"\n🏆 Best performing model: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['final_accuracy']:.4f}")
        print(f"   Training time: {best_model['training_time']:.1f}s")

        # Performance difference
        if len(results) == 2:
            acc_diff = abs(results[0]["final_accuracy"] - results[1]["final_accuracy"])
            time_diff = abs(results[0]["training_time"] - results[1]["training_time"])
            print(f"\n📊 Performance difference:")
            print(f"   Accuracy: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
            print(f"   Training time: {time_diff:.1f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    import json

    comparison_results = {
        "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "results": [
            {k: v for k, v in r.items() if k not in ["predictions", "true_labels"]}
            for r in results
        ],
    }

    output_path = f"{args.output_dir}/model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)

    logger.info(f"Comparison results saved to {output_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare mDeBERTa-v3 and BERT Japanese for toxicity classification"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/paired_inspection_ai_binary.csv",
        help="Path to the paired CSV data file",
    )

    parser.add_argument(
        "--use-romaji",
        action="store_true",
        help="Use romanized text instead of native Japanese",
    )

    parser.add_argument(
        "--sample-size", type=int, help="Limit training data size for comparison"
    )

    parser.add_argument(
        "--quick-test", action="store_true", help="Quick comparison with small dataset"
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )

    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size fraction"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/comparison",
        help="Directory to save comparison results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check data file
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return

    logger.info("Starting model comparison...")
    logger.info(f"Comparing: {list(MODELS.keys())}")

    results = compare_models(args)

    print(f"\n✅ Model comparison completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
