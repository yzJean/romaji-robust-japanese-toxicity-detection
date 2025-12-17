#!/usr/bin/env python3
"""
ByT5 Inference Script
Loads trained ByT5 model and performs inference with optional benchmarking.

Usage:
    python3 src/inference_byt5.py --model outputs/google_byt5_small_best_model.pt --text "テキスト"
    python3 src/inference_byt5.py --model outputs/google_byt5_small_romaji_best_model.pt --benchmark
"""

import argparse
import torch
import json
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from transformers import AutoTokenizer, T5ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_byt5_model(model_path, device='cuda'):
    """Load trained ByT5 model."""
    logger.info(f"Loading ByT5 model from {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model = T5ForSequenceClassification.from_pretrained('google/byt5-small', num_labels=2)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle checkpoint format - extract model_state_dict if it exists
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Fix nested "transformer.transformer" prefix and "transformer.classification_head"
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # Remove extra "transformer." prefix if it exists
        new_key = key.replace('transformer.transformer.', 'transformer.')
        new_key = new_key.replace('transformer.classification_head.', 'classification_head.')
        fixed_state_dict[new_key] = value
    
    model.load_state_dict(fixed_state_dict)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    
    logger.info(f"Model loaded successfully")
    return model, tokenizer, device


def predict_text(model, tokenizer, text, device, max_length=192):
    """
    Predict toxicity for a single text.
    Returns: (prediction, confidence, logits)
    """
    inputs = tokenizer(text, max_length=max_length, padding='max_length',
                      truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()
    
    return prediction, confidence, logits.cpu().numpy()


def benchmark_inference(model, tokenizer, texts, device, max_length=192, num_runs=3):
    """
    Benchmark inference performance on a batch of texts.
    
    Returns:
        dict with latency and memory metrics
    """
    logger.info(f"\nBenchmarking inference on {len(texts)} samples, {num_runs} runs...")
    
    model.eval()
    latencies = []
    gpu_memory_usage = []
    
    # Warm up
    logger.info("Warming up...")
    with torch.no_grad():
        for text in texts[:5]:
            inputs = tokenizer(text, max_length=max_length, padding='max_length',
                             truncation=True, return_tensors='pt').to(device)
            _ = model(**inputs)
    
    # Benchmark
    logger.info("Measuring inference latency...")
    for run in range(num_runs):
        run_latencies = []
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, max_length=max_length, padding='max_length',
                                 truncation=True, return_tensors='pt').to(device)
                
                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                _ = model(**inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                run_latencies.append(latency_ms)
        
        latencies.extend(run_latencies)
        
        # Record GPU memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_memory_usage.append(peak_memory_mb)
        
        logger.info(f"  Run {run+1}/{num_runs}: Mean latency = {np.mean(run_latencies):.2f}ms")
    
    # Statistics
    latencies = np.array(latencies)
    
    metrics = {
        'num_samples': len(texts) * num_runs,
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_median_ms': float(np.median(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_95th_ms': float(np.percentile(latencies, 95)),
        'latency_99th_ms': float(np.percentile(latencies, 99)),
        'gpu_memory_mb': float(np.mean(gpu_memory_usage)) if gpu_memory_usage else 0,
        'throughput_samples_per_sec': float(len(texts) * num_runs / (np.sum(latencies) / 1000)),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='ByT5 Inference Script')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained ByT5 model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to classify (optional)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmarking')
    parser.add_argument('--data-path', type=str, default='data/processed/paired_native_romaji_llmjp_binary.csv',
                       help='Path to test data for benchmarking')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to use for benchmarking')
    parser.add_argument('--max-length', type=int, default=192,
                       help='Max sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer, device = load_byt5_model(args.model, device)
    
    # Single text inference
    if args.text:
        logger.info(f"\nInferring on text: {args.text[:50]}...")
        pred, conf, logits = predict_text(model, tokenizer, args.text, device, args.max_length)
        
        label = 'Toxic' if pred == 1 else 'Non-Toxic'
        logger.info(f"Prediction: {label}")
        logger.info(f"Confidence: {conf:.4f}")
        logger.info(f"Logits: Non-Toxic={logits[0]:.4f}, Toxic={logits[1]:.4f}")
    
    # Benchmarking
    if args.benchmark:
        logger.info(f"\nLoading {args.num_samples} samples for benchmarking...")
        
        # Load test data
        df = pd.read_csv(args.data_path)
        np.random.seed(42)
        sample_indices = np.random.choice(len(df), size=min(args.num_samples, len(df)), replace=False)
        
        # Use romaji if available, else native
        if 'text_romaji' in df.columns:
            texts = df.iloc[sample_indices]['text_romaji'].values.tolist()
            logger.info("Using romaji text")
        else:
            texts = df.iloc[sample_indices]['text_native'].values.tolist()
            logger.info("Using native text")
        
        # Run benchmark
        metrics = benchmark_inference(model, tokenizer, texts, device, args.max_length, num_runs=3)
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("INFERENCE BENCHMARK RESULTS")
        logger.info("="*80)
        logger.info(f"Model: {args.model}")
        logger.info(f"Samples: {metrics['num_samples']}")
        logger.info(f"\nLatency Metrics:")
        logger.info(f"  Mean:        {metrics['latency_mean_ms']:.2f}ms")
        logger.info(f"  Median:      {metrics['latency_median_ms']:.2f}ms")
        logger.info(f"  Std Dev:     {metrics['latency_std_ms']:.2f}ms")
        logger.info(f"  95th %ile:   {metrics['latency_95th_ms']:.2f}ms")
        logger.info(f"  99th %ile:   {metrics['latency_99th_ms']:.2f}ms")
        logger.info(f"  Min:         {metrics['latency_min_ms']:.2f}ms")
        logger.info(f"  Max:         {metrics['latency_max_ms']:.2f}ms")
        logger.info(f"\nResource Usage:")
        logger.info(f"  GPU Memory:  {metrics['gpu_memory_mb']:.0f}MB")
        logger.info(f"  Throughput:  {metrics['throughput_samples_per_sec']:.1f} samples/sec")
        logger.info("="*80)
        
        # Save metrics
        output_file = 'outputs/byt5_inference_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {output_file}")


if __name__ == '__main__':
    main()
