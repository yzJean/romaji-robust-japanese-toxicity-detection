#!/usr/bin/env python3
"""
Simple training script for Japanese toxicity classification.
Quick verification of model training flow.
Supports microsoft/mdeberta-v3-base, tohoku-nlp/bert-base-japanese-v3, and google/byt5-small.

Usage:
    python3 src/train.py --quick-test                         # Quick verification: 50 samples, 1 epoch
    python3 src/train.py --model-type bert-japanese --quick-test  # Quick test with BERT Japanese
    python3 src/train.py --model-type mdeberta                 # Use mDeBERTa-v3 model
    python3 src/train.py --model-type bert-japanese            # Use BERT Japanese model
    python3 src/train.py --model-type byt5                     # Use ByT5-small byte-level model
    python3 src/train.py --sample-size 100                     # Use only 100 training samples
    python3 src/train.py --sample-size 200 --epochs 2          # 200 samples, 2 epochs
    python3 src/train.py --use-romaji                          # Full dataset with romaji
    python3 src/train.py --epochs 5 --batch-size 32            # Full training
"""

import argparse
import os
import sys

# If the user requested deterministic behavior via --deterministic we must set
# the cuBLAS workspace config BEFORE CUDA (and thus PyTorch) initializes.
# Check for the flag early on the raw argv and set the env var accordingly.
if "--deterministic" in sys.argv:
    # Use a safe workspace configuration recommended by PyTorch/NVIDIA docs
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import random
import logging

# Use CuPy if CUDA is available, otherwise fall back to NumPy
try:
    import cupy as cp
    if torch.cuda.is_available():
        np = cp
        print("Using CuPy for GPU-accelerated array operations")
    else:
        import numpy as np
        print("CUDA not available, using NumPy")
except ImportError:
    import numpy as np
    print("CuPy not installed, using NumPy")

from utils import load_data, SimpleToxicityDataset, SimpleBertClassifier, SimpleTrainer
import sentencepiece
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train simple BERT toxicity classifier"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/paired_native_romaji_inspection_ai_binary.csv",
        help="Path to the paired CSV data file",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mdeberta", "bert-japanese", "byt5"],
        help="Quick model selection: mdeberta, bert-japanese, or byt5. mdeberta is shortcut for microsoft/mdeberta-v3-base. bert-japanese is shortcut for tohoku-nlp/bert-base-japanese-v3. byt5 is shortcut for google/byt5-small (byte-level, tokenizer-free).",
    )

    parser.add_argument(
        "--use-romaji",
        action="store_true",
        help="Use romanized text instead of native Japanese",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )

    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, 
        help="Number of gradient accumulation steps (simulates larger batch size)"
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )

    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model and results",
    )

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit training data to this many samples for quick verification (e.g., --sample-size 100)",
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: use only 50 samples, 1 epoch, smaller batch size",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (may require setting CUBLAS_WORKSPACE_CONFIG).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set seeds for reproducibility
    seed = int(args.seed)
    logger.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic only when requested (may impact performance)
    if args.deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        # Prefer deterministic algorithms when available (may raise on some ops)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this API
            pass

    # Model selection logic
    if args.model_type == "mdeberta":
        args.model_name = "microsoft/mdeberta-v3-base"
        logger.info("Selected mDeBERTa-v3 model")
    elif args.model_type == "bert-japanese":
        args.model_name = "tohoku-nlp/bert-base-japanese-v3"
        logger.info("Selected BERT Japanese model")
    elif args.model_type == "byt5":
        args.model_name = "google/byt5-small"
        logger.info("Selected ByT5-small byte-level model")
        # ByT5 requires longer training due to byte-level processing
        if args.epochs == 3:  # Only adjust if using default epochs
            args.epochs = 10  # Increased from 8 for better convergence
            logger.info("ByT5 detected: increasing epochs to 10 (byte-level models need longer training)")
        # Reduce batch size and max_length for ByT5 to fit in GPU memory
        if args.batch_size == 16:
            args.batch_size = 8  # Increased from 4
            args.gradient_accumulation_steps = 2  # Effective batch size 16
            logger.info("ByT5 detected: batch size 8 with 2 gradient accumulation steps")
        # Reduce max_length for ByT5 (byte-level is more memory intensive)
        if args.max_length == 512:
            args.max_length = 256
            logger.info("ByT5 detected: reducing max_length to 256 bytes (sufficient for most Japanese text)")
    else:
        # Default to mDeBERTa if no model specified
        args.model_name = "microsoft/mdeberta-v3-base"
        logger.info("Selected mDeBERTa-v3 model (default)")

    logger.info(f"Using model: {args.model_name}")

    # Check data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        logger.error("Please ensure you have the processed data file.")
        return

    # Quick test mode adjustments
    if args.quick_test:
        args.sample_size = 50
        args.epochs = 1
        args.batch_size = 8
        logger.info("Quick test mode activated: 50 samples, 1 epoch, batch size 8")

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

        # Also limit test size proportionally
        test_sample_size = max(10, args.sample_size // 4)  # At least 10 test samples
        test_texts = test_texts[:test_sample_size]
        test_labels = test_labels[:test_sample_size]
        # print(f"test_texts: {test_texts}")
        # print(f"test_labels: {test_labels}")
        logger.info(
            f"Data size limited: {original_size} → {len(train_texts)} train samples"
        )
        logger.info(f"Test samples: {len(test_texts)}")
        logger.info(
            f"New label distribution - Train: {dict(zip(*np.unique(train_labels, return_counts=True)))}"
        )
        logger.info(
            f"New label distribution - Test: {dict(zip(*np.unique(test_labels, return_counts=True)))}"
        )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = SimpleToxicityDataset(
        train_texts, train_labels, tokenizer, args.max_length
    )
    test_dataset = SimpleToxicityDataset(
        test_texts, test_labels, tokenizer, args.max_length
    )

    # Create deterministic generator for DataLoader shuffling
    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    # Create dataloaders (pass generator to ensure deterministic shuffling)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=dl_generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, generator=dl_generator
    )

    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Calculate class weights to handle imbalanced datasets
    from sklearn.utils.class_weight import compute_class_weight
    
    # Convert to NumPy for sklearn compatibility
    if np.__name__ == 'cupy':
        train_labels_np = np.asnumpy(train_labels) if hasattr(train_labels, '__array_interface__') else np.array(train_labels).get()
    else:
        train_labels_np = np.array(train_labels)
    
    unique_labels = np.unique(train_labels_np) if np.__name__ != 'cupy' else __import__('numpy').unique(train_labels_np)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels_np)
    
    # Get class distribution using the appropriate numpy
    if np.__name__ == 'cupy':
        import numpy as numpy_lib
        counts = numpy_lib.unique(train_labels_np, return_counts=True)
    else:
        counts = np.unique(train_labels_np, return_counts=True)
    
    logger.info(f"Class distribution: {dict(zip(*counts))}")
    logger.info(f"Computed class weights: {dict(zip(unique_labels, class_weights))}")

    # Create model
    logger.info("Creating BERT model...")
    # Use float32 for deterministic mode or ByT5 to avoid NaN issues with mixed precision
    # ByT5 in float16 can have numerical instability with weighted loss
    use_float32 = args.deterministic or args.model_type == "byt5"
    if use_float32:
        if args.deterministic:
            logger.info(
                "Deterministic mode enabled: using float32 to avoid numerical instability"
            )
        if args.model_type == "byt5":
            logger.info(
                "ByT5 model: using float32 to avoid NaN loss with weighted loss function"
            )
    model = SimpleBertClassifier(
        args.model_name, dropout=args.dropout, use_float32=use_float32
    )

    # Lower learning rate for ByT5 to prevent instability
    learning_rate = args.learning_rate
    if args.model_type == "byt5" and learning_rate == 2e-5:
        learning_rate = 1e-5  # Adjusted from 5e-6 for better convergence
        logger.info(f"ByT5 detected: setting learning rate to {learning_rate} for optimal convergence")

    # Create trainer with class weights and gradient clipping
    trainer = SimpleTrainer(
        model, device, learning_rate, 
        class_weights=class_weights, 
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)

        # Evaluate
        val_loss, val_acc, _, _ = trainer.evaluate(test_loader)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create safe filename from model name
            safe_model_name = args.model_name.replace("/", "_").replace("-", "_")
            # Add romaji tag if using romanized text
            romaji_tag = "_romaji" if args.use_romaji else ""
            model_path = os.path.join(
                args.output_dir, f"{safe_model_name}{romaji_tag}_best_model.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_name": args.model_name,
                    "use_romaji": args.use_romaji,
                    "config": vars(args),
                    "val_acc": val_acc,
                },
                model_path,
            )
            logger.info(f"New best model saved with val_acc: {val_acc:.4f}")

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    test_loss, test_acc, predictions, true_labels = trainer.evaluate(test_loader)

    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Detailed metrics
    print("\nClassification Report:")
    report = classification_report(
        true_labels, predictions, target_names=["Non-Toxic", "Toxic"], zero_division=0
    )
    print(report)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)

    # Convert confusion matrix to list for JSON serialization
    if hasattr(cm, 'get'):
        # CuPy array
        cm_list = cm.get().tolist()
    else:
        # NumPy array
        cm_list = cm.tolist()

    # Save results
    results = {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "classification_report": report,
        "confusion_matrix": cm_list,
        "config": vars(args),
    }

    import json

    # Create safe filename from model name
    safe_model_name = args.model_type
    # Add romaji tag if using romanized text
    romaji_tag = "_romaji" if args.use_romaji else ""
    results_path = os.path.join(
        args.output_dir, f"{safe_model_name}{romaji_tag}_results.json"
    )
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()
        }
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_path}")

    # Save configuration
    config_path = os.path.join(
        args.output_dir, f"{safe_model_name}{romaji_tag}_config.json"
    )
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Configuration saved to {config_path}")

    logger.info(
        f"\nTraining completed! Check {args.output_dir} for saved model and results."
    )


if __name__ == "__main__":
    main()
