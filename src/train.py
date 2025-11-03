#!/usr/bin/env python3
"""
Simple training script for BERT toxicity classification.
Quick verification of model training flow.

Usage:
    python3 train.py --quick-test                    # Quick verification: 50 samples, 1 epoch
    python3 train.py --xlm-roberta --quick-test      # Quick test with XLM-RoBERTa
    python3 train.py --model-type bert               # Use BERT model
    python3 train.py --model-type xlm-roberta        # Use XLM-RoBERTa model
    python3 train.py --sample-size 100               # Use only 100 training samples  
    python3 train.py --sample-size 200 --epochs 2    # 200 samples, 2 epochs
    python3 train.py --use-romaji                     # Full dataset with romaji
    python3 train.py --epochs 5 --batch-size 32      # Full training
"""

import argparse
import torch
import logging
import numpy as np
from utils import (
    load_data, SimpleToxicityDataset, SimpleBertClassifier, SimpleTrainer
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train simple BERT toxicity classifier')
    
    parser.add_argument('--data-path', type=str, 
                       default='data/processed/paired_inspection_ai_binary.csv',
                       help='Path to the paired CSV data file')

    parser.add_argument('--model-type', type=str, choices=['bert', 'xlm-roberta'],
                       help='Quick model selection: bert or xlm-roberta')
    
    parser.add_argument('--xlm-roberta', action='store_true',
                       help='Use XLM-RoBERTa model (shortcut for FacebookAI/xlm-roberta-base)')
    
    parser.add_argument('--use-romaji', action='store_true',
                       help='Use romanized text instead of native Japanese')
    
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save model and results')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    parser.add_argument('--sample-size', type=int,
                       help='Limit training data to this many samples for quick verification (e.g., --sample-size 100)')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: use only 50 samples, 1 epoch, smaller batch size')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Model selection logic
    if args.xlm_roberta or args.model_type == 'xlm-roberta':
        args.model_name = 'FacebookAI/xlm-roberta-base'
        logger.info("Selected XLM-RoBERTa model")
    elif args.model_type == 'bert':
        args.model_name = 'google-bert/bert-base-multilingual-cased'
        logger.info("Selected BERT model")
    else:
        # Default to BERT if no model specified
        args.model_name = 'google-bert/bert-base-multilingual-cased'
        logger.info("Selected BERT model (default)")
    
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
        train_texts = train_texts[:args.sample_size]
        train_labels = train_labels[:args.sample_size]
        
        # Also limit test size proportionally
        test_sample_size = max(10, args.sample_size // 4)  # At least 10 test samples
        test_texts = test_texts[:test_sample_size] 
        test_labels = test_labels[:test_sample_size]
        print(f"test_texts: {test_texts}")
        print(f"test_labels: {test_labels}")
        logger.info(f"Data size limited: {original_size} → {len(train_texts)} train samples")
        logger.info(f"Test samples: {len(test_texts)}")
        logger.info(f"New label distribution - Train: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        logger.info(f"New label distribution - Test: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = SimpleToxicityDataset(train_texts, train_labels, tokenizer, args.max_length)
    test_dataset = SimpleToxicityDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating BERT model...")
    model = SimpleBertClassifier(args.model_name, dropout=args.dropout)
    
    # Create trainer
    trainer = SimpleTrainer(model, device, args.learning_rate)
    
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
            safe_model_name = args.model_name.replace('/', '_').replace('-', '_')
            model_path = os.path.join(args.output_dir, f'{safe_model_name}_best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': args.model_name,
                'use_romaji': args.use_romaji,
                'config': vars(args),
                'val_acc': val_acc
            }, model_path)
            logger.info(f"New best model saved with val_acc: {val_acc:.4f}")
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    
    test_loss, test_acc, predictions, true_labels = trainer.evaluate(test_loader)
    
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Detailed metrics
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=['Non-Toxic', 'Toxic'], zero_division=0)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'config': vars(args)
    }
    
    import json
    # Create safe filename from model name
    safe_model_name = args.model_name.replace('/', '_').replace('-', '_')
    results_path = os.path.join(args.output_dir, f'{safe_model_name}_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()}
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_path}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, f'{safe_model_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")
    
    logger.info(f"\nTraining completed! Check {args.output_dir} for saved model and results.")


if __name__ == '__main__':
    main()