#!/usr/bin/env python3
"""
Inference script for transformer-based toxicity classification.
Runs model-by-model evaluation and produces CSV results for Section 7.2.3 analysis.

Usage:
    python inference.py --model outputs/model.pt --output results.csv
    python inference.py --model outputs/model.pt --text "Hello world"
"""

import argparse
import torch
import os
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from utils import SimpleBertClassifier, predict_text
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device):
    """Load a saved model."""
    try:
        # Try loading with weights_only=False for compatibility with older PyTorch saves
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load model with weights_only=False: {e}")
        # If that fails, try the safe loading approach
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e2:
            logger.error(f"Failed to load model with weights_only=True: {e2}")
            raise e2

    # Get model configuration
    tokenizer_name = checkpoint["tokenizer_name"]
    use_romaji = checkpoint.get("use_romaji", False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create model
    model = SimpleBertClassifier(tokenizer_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Uses romaji: {use_romaji}")

    return model, tokenizer, use_romaji


def evaluate_model_with_paired_data(model, tokenizer, device, data_path, output_path):
    """
    Evaluate model on paired native/romaji test data and save results to CSV.

    Schema: NativeJapanese, Romaji, ToxicityGT, Results, IsTruePositive

    Args:
        model: Loaded model
        tokenizer: Model's tokenizer
        device: PyTorch device
        data_path: Path to paired dataset CSV
        output_path: Where to save the results CSV
    """
    from sklearn.metrics import accuracy_score, classification_report

    logger.info("=" * 80)
    logger.info("MODEL EVALUATION ON PAIRED NATIVE/ROMAJI DATA")
    logger.info("=" * 80)

    # Load paired test data
    df = pd.read_csv(data_path)

    # Split to get test set (same split as training)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_int_coarse"]
    )

    logger.info(f"Loaded {len(test_df)} test samples from {data_path}")
    logger.info(
        f"Toxic: {sum(test_df['label_int_coarse'])}, Non-toxic: {len(test_df) - sum(test_df['label_int_coarse'])}"
    )

    # Get model info from checkpoint to determine if it uses romaji
    # We'll assume romaji models should use romaji text
    # This can be detected from the model's use_romaji flag if stored in checkpoint

    # Run predictions
    results_data = []
    predictions = []

    logger.info("\nRunning predictions...")

    for idx, row in test_df.iterrows():
        native_text = row["text_native"]
        romaji_text = row["text_romaji"]
        ground_truth = int(row["label_int_coarse"])

        # Use romaji text for prediction (since you want romaji models)
        result = predict_text(model, tokenizer, romaji_text, device)
        pred = 1 if result["prediction"] == "Toxic" else 0
        predictions.append(pred)

        # Determine if prediction is correct
        is_true_positive = pred == ground_truth

        # Add to results
        results_data.append(
            {
                "NativeJapanese": native_text,
                "Romaji": romaji_text,
                "ToxicityGT": ground_truth,
                "Results": pred,
                "IsTruePositive": is_true_positive,
            }
        )

    # Create DataFrame with specified schema
    results_df = pd.DataFrame(results_data)

    # Calculate and display metrics
    ground_truth_labels = test_df["label_int_coarse"].tolist()
    accuracy = accuracy_score(ground_truth_labels, predictions)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(
        f"Correct predictions: {sum(results_df['IsTruePositive'])}/{len(results_df)}"
    )

    logger.info("\nClassification Report:")
    print(
        classification_report(
            ground_truth_labels,
            predictions,
            target_names=["Non-Toxic", "Toxic"],
            zero_division=0,
        )
    )

    # Save results
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Results saved to {output_path}")
    logger.info(f"Schema: NativeJapanese, Romaji, ToxicityGT, Results, IsTruePositive")

    return results_df, accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with trained transformer toxicity classifier"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model file (.pt)"
    )

    parser.add_argument("--text", type=str, help="Single text to classify")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/paired_native_romaji_llmjp_binary.csv",
        help="Path to paired data file for evaluation",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file to save results (required for evaluation mode)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    # Load model
    model, tokenizer, use_romaji = load_model(args.model, device)

    # Single text inference
    if args.text:
        logger.info(f"\nClassifying single text: '{args.text}'")
        result = predict_text(model, tokenizer, args.text, device)

        print(f"\nResult:")
        print(f"  Text: {result['text']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Toxic Probability: {result['toxic_probability']:.3f}")

    # Evaluation mode - generate CSV with specified schema
    else:
        if not args.output:
            logger.error("Please provide --output path for CSV results")
            logger.info(
                "Usage: python inference.py --model <model.pt> --output <output.csv>"
            )
            return

        if not os.path.exists(args.data_path):
            logger.error(f"Data file not found: {args.data_path}")
            return

        # Run evaluation and save results
        evaluate_model_with_paired_data(
            model, tokenizer, device, args.data_path, args.output
        )


if __name__ == "__main__":
    main()
