#!/bin/bash
# Run inference on romaji-trained models for Section 7.2.3 Error Taxonomy
# This script runs both mDeBERTa and BERT models trained on romaji data

set -e  # Exit on error

echo "=========================================="
echo "Section 7.2.3 Error Taxonomy Analysis"
echo "Running romaji-trained models"
echo "=========================================="

# Data path
DATA_PATH="data/processed/paired_native_romaji_llmjp_binary.csv"

# Output directory
OUTPUT_DIR="outputs/eval"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "1. Running mDeBERTa (romaji) model..."
python src/inference.py \
  --model outputs/training_romaji/mdeberta_data_romaji_lr_20_15_v/microsoft_mdeberta_v3_base_romaji_best_model.pt \
  --data-path "$DATA_PATH" \
  --output "$OUTPUT_DIR/mdeberta_romaji_results.csv"

echo ""
echo "2. Running BERT Japanese (romaji) model..."
python src/inference.py \
  --model outputs/training_romaji/bert_data_romaji_lr_20_15_v/tohoku_nlp_bert_base_japanese_v3_romaji_best_model.pt \
  --data-path "$DATA_PATH" \
  --output "$OUTPUT_DIR/bert_romaji_results.csv"

echo ""
echo "=========================================="
echo "✓ Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/mdeberta_romaji_results.csv"
echo "  - $OUTPUT_DIR/bert_romaji_results.csv"
echo ""
echo "CSV Schema: NativeJapanese, Romaji, ToxicityGT, Results, IsTruePositive"
echo ""
