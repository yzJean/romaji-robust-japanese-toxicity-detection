#!/usr/bin/env python3
"""
Simple inference script for transformer-based toxicity classification.
Supports mDeBERTa-v3 and BERT Japanese models.
Test trained model on new texts or evaluate on test data.

Usage:
    python inference.py --model outputs/best_model.pt
    python inference.py --model outputs/best_model.pt --text "Hello world"
    python inference.py --model outputs/best_model.pt --evaluate
"""

import argparse
import torch
import json
import os
from transformers import AutoTokenizer
from utils import SimpleBertClassifier, predict_text, load_data, SimpleToxicityDataset
from torch.utils.data import DataLoader
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


def interactive_inference(model, tokenizer, device):
    """Interactive inference mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TOXICITY DETECTION")
    print("=" * 60)
    print("Enter Japanese text to classify (or 'quit' to exit)")
    print("Examples:")
    print("  ありがとう (Thank you)")
    print("  バカ野郎 (Stupid bastard)")
    print("-" * 60)

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() in ["quit", "exit", "q"]:
            break

        if not text:
            continue

        try:
            result = predict_text(model, tokenizer, text, device)

            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Toxic Probability: {result['toxic_probability']:.3f}")

            # Color coding for terminal
            if result["prediction"] == "Toxic":
                status = "🚨 TOXIC"
            else:
                status = "✅ NON-TOXIC"

            print(f"Status: {status}")

        except Exception as e:
            print(f"Error processing text: {e}")


def batch_inference(model, tokenizer, device, texts):
    """Run inference on a batch of texts."""
    results = []

    print(f"\nRunning inference on {len(texts)} texts...")

    for text in texts:
        try:
            result = predict_text(model, tokenizer, text, device)
            results.append(result)

            print(f"Text: {result['text'][:50]}...")
            print(f"  → {result['prediction']} (conf: {result['confidence']:.3f})")

        except Exception as e:
            print(f"Error processing '{text[:30]}...': {e}")
            results.append(
                {
                    "text": text,
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "toxic_probability": 0.0,
                }
            )

    return results


def evaluate_on_test_data(model, tokenizer, device, data_path, use_romaji=False):
    """Evaluate model on test data."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Load test data
    _, test_texts, _, test_labels = load_data(
        data_path, use_romaji=use_romaji, test_size=0.2
    )

    logger.info(f"Evaluating on {len(test_texts)} test samples...")

    # Get predictions
    predictions = []
    probabilities = []

    for text in test_texts:
        result = predict_text(model, tokenizer, text, device)
        pred = 1 if result["prediction"] == "Toxic" else 0
        predictions.append(pred)
        probabilities.append(result["toxic_probability"])

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=["Non-Toxic", "Toxic"],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "probabilities": probabilities,
        "true_labels": test_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with trained transformer toxicity classifier"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model file (.pt)"
    )

    parser.add_argument("--text", type=str, help="Single text to classify")

    parser.add_argument(
        "--texts-file",
        type=str,
        help="File containing texts to classify (one per line)",
    )

    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test data")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/paired_inspection_ai_binary.csv",
        help="Path to data file for evaluation",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument(
        "--output", type=str, help="Output file to save results (JSON format)"
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

    results = {}

    # Single text inference
    if args.text:
        print(f"\nClassifying single text: '{args.text}'")
        result = predict_text(model, tokenizer, args.text, device)

        print(f"\nResult:")
        print(f"  Text: {result['text']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Toxic Probability: {result['toxic_probability']:.3f}")

        results["single_text"] = result

    # Batch inference from file
    elif args.texts_file:
        if not os.path.exists(args.texts_file):
            logger.error(f"Texts file not found: {args.texts_file}")
            return

        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        batch_results = batch_inference(model, tokenizer, device, texts)
        results["batch_results"] = batch_results

    # Evaluation mode
    elif args.evaluate:
        if not os.path.exists(args.data_path):
            logger.error(f"Data file not found: {args.data_path}")
            return

        eval_results = evaluate_on_test_data(
            model, tokenizer, device, args.data_path, use_romaji
        )
        results["evaluation"] = eval_results

    # Interactive mode
    elif args.interactive:
        interactive_inference(model, tokenizer, device)
        return

    # Default: run some sample texts
    else:
        print("\nNo specific mode selected. Running sample texts...")

        sample_texts = [
            "ありがとう",  # Thank you
            "こんにちは",  # Hello
            "バカ野郎",  # Stupid bastard
            "死ね",  # Die
            "いい天気ですね",  # Nice weather
            "クソ野郎",  # Damn bastard
            "愛してる",  # I love you
            "殺すぞ",  # I'll kill you
        ]

        batch_results = batch_inference(model, tokenizer, device, sample_texts)
        results["sample_results"] = batch_results

    # Save results if requested
    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
