#!/usr/bin/env python3
"""
Inference script for transformer-based toxicity classification.
Runs model-by-model evaluation and produces CSV results for Section 7.2.3 analysis.

Usage:
    python inference.py --model outputs/model.pt --output results.csv
    python inference.py --model outputs/model.pt --text "Hello world"
"""

import argparse
import json
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


def evaluate_model_with_paired_data(
    model,
    tokenizer,
    device,
    data_path,
    output_path,
    language="romaji",
    report_output=None,
):
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

        # Choose which text to feed the model
        input_text = romaji_text if language == "romaji" else native_text

        result = predict_text(model, tokenizer, input_text, device)
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
    # Print human-readable report
    report_text = classification_report(
        ground_truth_labels,
        predictions,
        target_names=["Non-Toxic", "Toxic"],
        zero_division=0,
    )
    print(report_text)

    # Also get a serializable dict for saving
    report_dict = classification_report(
        ground_truth_labels,
        predictions,
        target_names=["Non-Toxic", "Toxic"],
        output_dict=True,
        zero_division=0,
    )

    # Save results CSV
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Results saved to {output_path}")
    logger.info(f"Schema: NativeJapanese, Romaji, ToxicityGT, Results, IsTruePositive")

    # Save classification report and summary if requested
    if report_output:
        report_dir = os.path.dirname(report_output)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)

        # Write text version (same as printed)
        txt_path = (
            report_output if report_output.endswith(".txt") else report_output + ".txt"
        )
        with open(txt_path, "w") as ftxt:
            ftxt.write("Accuracy: {:.6f}\n".format(accuracy))
            ftxt.write(
                "Correct: {}/{}\n\n".format(
                    sum(results_df["IsTruePositive"]), len(results_df)
                )
            )
            ftxt.write(report_text)

        # Write JSON version for programmatic use
        json_path = (
            report_output + ".json"
            if not report_output.endswith(".json")
            else report_output
        )
        summary = {
            "accuracy": float(accuracy),
            "correct": int(sum(results_df["IsTruePositive"])),
            "total": int(len(results_df)),
            "classification_report": report_dict,
            "language": language,
        }
        with open(json_path, "w") as fj:
            json.dump(summary, fj, indent=2, ensure_ascii=False)

        logger.info(f"Saved classification report to {txt_path} and {json_path}")

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

    parser.add_argument(
        "--language",
        type=str,
        choices=["native", "romaji"],
        default="romaji",
        help="Which language text to use for prediction (native or romaji)",
    )

    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Optional path (prefix) to write classification report (.txt and .json will be written)",
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
            model,
            tokenizer,
            device,
            args.data_path,
            args.output,
            language=args.language,
            report_output=args.report_output,
        )


if __name__ == "__main__":
    main()
