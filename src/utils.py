"""
Transformer-based Japanese toxicity classification utilities.
Supports mDeBERTa-v3 and BERT Japanese models for quick verification of training and inference flow.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import os
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleToxicityDataset(Dataset):
    """Simple PyTorch Dataset for Japanese toxicity classification."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SimpleBertClassifier(nn.Module):
    """Simple transformer classifier for binary toxicity detection.
    Supports encoder-only models (mDeBERTa-v3, BERT Japanese) and encoder-decoder models (ByT5)."""

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model_name = model_name
        self.is_t5 = "t5" in model_name.lower()

        # Loading large transformer weights can momentarily spike CPU memory usage.
        # low_cpu_mem_usage streams the weights in and keeps peak memory low. We
        # fall back gracefully if this transformers version does not support it.
        load_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }

        # T5 models need special handling - use AutoModelForSequenceClassification
        if self.is_t5:
            load_kwargs["num_labels"] = 2
            try:
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **load_kwargs
                )
            except TypeError:
                load_kwargs.pop("low_cpu_mem_usage", None)
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **load_kwargs
                )
            except RuntimeError:
                load_kwargs["torch_dtype"] = torch.float32
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **load_kwargs
                )
            self.dropout = None
            self.classifier = None
        else:
            # Encoder-only models (BERT, DeBERTa, etc.)
            try:
                self.transformer = AutoModel.from_pretrained(model_name, **load_kwargs)
            except TypeError:
                # Older transformers releases do not accept low_cpu_mem_usage.
                load_kwargs.pop("low_cpu_mem_usage", None)
                self.transformer = AutoModel.from_pretrained(model_name, **load_kwargs)
            except RuntimeError:
                # Some CPU builds cannot handle float16 weights; retry with float32.
                load_kwargs["torch_dtype"] = torch.float32
                self.transformer = AutoModel.from_pretrained(model_name, **load_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(
                self.transformer.config.hidden_size, 2
            )  # Binary classification

        # Determine model type for logging
        if "mdeberta" in model_name.lower():
            model_type = "mDeBERTa-v3"
        elif "bert-base-japanese" in model_name.lower():
            model_type = "BERT Japanese"
        elif self.is_t5:
            model_type = "T5/ByT5"
        else:
            model_type = "Transformer"

        logger.info(f"{model_type} model initialized with {model_name}")
        if not self.is_t5:
            logger.info(f"Hidden size: {self.transformer.config.hidden_size}")

    def forward(self, input_ids, attention_mask):
        if self.is_t5:
            # T5 models handle classification internally
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits
        else:
            # Encoder-only models
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

            # Use [CLS] token representation
            pooled_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = self.dropout(pooled_output)

            logits = self.classifier(pooled_output)
            return logits


class SimpleTrainer:
    """Simple trainer for quick model verification."""

    def __init__(self, model, device, learning_rate: float = 2e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return total_loss / len(dataloader), accuracy, all_preds, all_labels


def load_data(
    csv_path: str, use_romaji: bool = False, test_size: float = 0.2
) -> Tuple[List, List, List, List]:
    """
    Load and split the paired CSV data.

    Args:
        csv_path: Path to the paired CSV file (e.g., paired_native_romaji_inspection_ai_binary.csv)
        use_romaji: Whether to use romanized text instead of native Japanese
        test_size: Fraction of data to use for testing

    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    # Select text column
    text_column = "text_romaji" if use_romaji else "text_native"
    if text_column not in df.columns:
        logger.warning(f"{text_column} not found, using text_native")
        text_column = "text_native"

    texts = df[text_column].tolist()
    labels = df["label_int_coarse"].tolist()  # Binary labels: 0=non-toxic, 1=toxic

    # Remove any rows with NaN labels (shouldn't happen in binary strict data)
    clean_data = [(t, l) for t, l in zip(texts, labels) if pd.notna(l)]
    texts, labels = zip(*clean_data) if clean_data else ([], [])

    logger.info(f"Using {len(texts)} samples after cleaning")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    logger.info(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

    return train_texts, test_texts, train_labels, test_labels


def main():
    """Main training and evaluation function."""

    # Configuration
    DATA_PATH = "data/processed/paired_native_romaji_inspection_ai_binary.csv"
    MODEL_NAME = "microsoft/mdeberta-v3-base"
    USE_ROMAJI = False  # Change to True to use romanized text
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 512

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_texts, test_texts, train_labels, test_labels = load_data(
        DATA_PATH, use_romaji=USE_ROMAJI
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    train_dataset = SimpleToxicityDataset(
        train_texts, train_labels, tokenizer, MAX_LENGTH
    )
    test_dataset = SimpleToxicityDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = SimpleBertClassifier(MODEL_NAME)

    # Create trainer
    trainer = SimpleTrainer(model, device, LEARNING_RATE)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc, _, _ = trainer.evaluate(test_loader)

        logger.info(f"Epoch {epoch+1}/{EPOCHS}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final evaluation
    logger.info("Final evaluation...")
    test_loss, test_acc, predictions, true_labels = trainer.evaluate(test_loader)

    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=["Non-Toxic", "Toxic"],
            zero_division=0,
        )
    )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

    # Save model
    model_path = f"simple_bert_toxicity_{'romaji' if USE_ROMAJI else 'native'}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer_name": MODEL_NAME,
            "use_romaji": USE_ROMAJI,
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path}")

    return model, tokenizer


def predict_text(model, tokenizer, text: str, device, max_length: int = 512):
    """
    Predict toxicity for a single text.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Input text
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Dictionary with prediction and confidence
    """
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)

    confidence = probabilities[0][prediction].item()
    pred_label = "Toxic" if prediction.item() == 1 else "Non-Toxic"

    return {
        "text": text,
        "prediction": pred_label,
        "confidence": confidence,
        "toxic_probability": probabilities[0][1].item(),
    }


if __name__ == "__main__":
    # Check if data exists
    data_path = "data/processed/paired_native_romaji_inspection_ai_binary.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please ensure you have run the data processing pipeline first.")
        exit(1)

    # Run training
    model, tokenizer = main()

    # Test inference with some examples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_texts = [
        "ありがとう",  # Thank you
        "バカ野郎",  # Stupid bastard
        "いい天気ですね",  # Nice weather
        "死ね",  # Die
    ]

    print("\n" + "=" * 50)
    print("Testing inference on sample texts:")
    print("=" * 50)

    for text in test_texts:
        result = predict_text(model, tokenizer, text, device)
        print(f"Text: {result['text']}")
        print(
            f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})"
        )
        print(f"Toxic probability: {result['toxic_probability']:.3f}")
        print("-" * 30)
