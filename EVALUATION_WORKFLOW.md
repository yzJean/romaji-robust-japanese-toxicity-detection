# Model Evaluation and Analysis Workflow Guide

This guide covers the complete evaluation workflow for Section 7.2.3 (Error Taxonomy Analysis), from training models to computing error statistics.

## Table of Contents
1. [Training Models](#1-training-models)
2. [Tokenization Diagnostics](#2-tokenization-diagnostics-section-721)
3. [Model Inference (Individual)](#3-model-inference-individual-runs)
4. [Batch Inference (Taxonomy Data)](#4-batch-inference-for-taxonomy-analysis)
5. [Error Taxonomy Statistics](#5-error-taxonomy-statistics-section-723)

---

## 1. Training Models

### Train All Tokenizer Models at Once

Use the `run_tokenizer_models.sh` script to train all tokenizer-based models (BERT Japanese and mDeBERTa) on both native and romaji text:

```bash
# Quick test mode (fast, for verification)
./scripts/run_tokenizer_models.sh

# Full training mode (complete experiments)
FULL=1 ./scripts/run_tokenizer_models.sh
```

**Models trained:**
- mDeBERTa (native script)
- mDeBERTa (romaji)
- BERT Japanese (native script)
- BERT Japanese (romaji)

**Output locations:**
```
outputs/<run_dir>
```

Each directory contains:
- `<model_name>_best_model.pt` - Best checkpoint
- `<model_name>_config.json` - Training configuration
- `<model_name>_results.json` - Training metrics

**Example directory:** `outputs/tokenizer_runs/run_2_model_type_mdeberta_data_path_..._use_romaji/`

---

## 2. Tokenization Diagnostics (Section 7.2.1)

After training, run tokenization diagnostics to analyze tokenization-specific artifacts on romaji text.

### Command Template

```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name <huggingface-model-name> \
  --checkpoint <path-to-best-model.pt> \
  --output outputs/eval/diagnostics_<model>_<native|romaji>.json
```

### Examples

#### mDeBERTa (Native)
```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name microsoft/mdeberta-v3-base \
  --checkpoint outputs/tokenizer_runs/run_1_model_type_mdeberta_data_path_data_processed_paired_native_romaji_llmjp_binary_csv/mdeberta_best_model.pt \
  --output outputs/eval/diagnostics_mdeberta_native.json
```

#### mDeBERTa (Romaji)
```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name microsoft/mdeberta-v3-base \
  --checkpoint outputs/training_romaji/mdeberta_data_romaji/microsoft_mdeberta_v3_base_romaji_best_model.pt \
  --output outputs/eval/diagnostics_mdeberta_romaji.json
```

#### BERT Japanese (Native)
```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name tohoku-nlp/bert-base-japanese-v3 \
  --checkpoint outputs/tokenizer_runs/bert_japanese_best_model.pt \
  --output outputs/eval/diagnostics_bert_japanese_native.json
```

#### BERT Japanese (Romaji)
```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name tohoku-nlp/bert-base-japanese-v3 \
  --checkpoint outputs/training_romaji/bert_data_romaji/bert_romajitrained_best_model.pt \
  --output outputs/eval/diagnostics_bert_japanese_romaji.json
```

### Output Format

Each diagnostic file contains:
- `avg_tokens_native` / `avg_tokens_romaji` - Average tokens per sentence
- `avg_token_ratio_romaji_over_native` - Tokenization granularity ratio
- `oov_rate_native` / `oov_rate_romaji` - Out-of-vocabulary rates
- `accuracy_native` / `accuracy_romaji` - Model accuracy on each script
- `flip_rate` - Percentage of predictions that change between scripts
- `mcnemar_p_value` - Statistical significance of performance difference
- `top_flip_examples` - Sample cases where predictions differ

**Key Metric:** `avg_token_ratio_romaji_over_native` > 1.0 indicates tokenization fragmentation on romaji.

---

## 3. Model Inference (Individual Runs)

Run inference on individual models to generate prediction results with the schema:
```
NativeJapanese, Romaji, ToxicityGT, Results, IsTruePositive
```

### Command Template

```bash
python src/inference.py \
  --model <path-to-model.pt> \
  --output <output-csv-path>
```

### Examples

#### mDeBERTa (Romaji)
```bash
python src/inference.py \
  --model outputs/training_romaji/mdeberta_data_romaji/microsoft_mdeberta_v3_base_romaji_best_model.pt \
  --output outputs/eval/mdeberta_romaji_results.csv
```

#### BERT Japanese (Romaji)
```bash
python src/inference.py \
  --model outputs/training_romaji/bert_data_romaji/bert_romajitrained_best_model.pt \
  --output outputs/eval/bert_romaji_results.csv
```

#### ByT5 (Romaji) - When Available
```bash
python src/inference.py \
  --model outputs/training_romaji/byt5_data_romaji_lr_20_15_v/google_byt5_small_romaji_best_model.pt \
  --output outputs/eval/byt5_romaji_results.csv
```

### Single Text Classification

You can also test individual texts:
```bash
python src/inference.py \
  --model <path-to-model.pt> \
  --text "bakayarou"
```

---

## 4. Batch Inference for Taxonomy Analysis

For convenience, use the batch script to run inference on multiple tokenizer models at once:

```bash
./scripts/run_taxonomy_analysis_tokenizer_model.sh
```

This script runs:
1. mDeBERTa (romaji) → `outputs/eval/mdeberta_romaji_results.csv`
2. BERT Japanese (romaji) → `outputs/eval/bert_romaji_results.csv`

**Note:** The script uses hardcoded model paths. If your models are in different locations, either:
- Update the paths in the script, or
- Run individual inference commands (see Section 3)

---

## 5. Error Taxonomy Statistics (Section 7.2.3)

Compute Type B / Type C ratio to understand error patterns.

### Error Categories

- **Type A**: Correct predictions (model got it right)
- **Type B**: False Negatives - Model MISSES toxic content (`ToxicityGT=1, Results=0`)
- **Type C**: False Positives - Model OVER-FLAGS non-toxic content (`ToxicityGT=0, Results=1`)

**Type B / Type C Ratio** = (False Negatives) / (False Positives)

### Using the Analysis Script

```bash
# Analyze specific models
./scripts/analyze_errors.sh <csv1> [csv2] [csv3] ...

# Examples:

# Two models (BERT + mDeBERTa)
./scripts/analyze_errors.sh \
    outputs/eval/bert_romaji_results.csv \
    outputs/eval/mdeberta_romaji_results.csv

# Three models (BERT + mDeBERTa + ByT5)
./scripts/analyze_errors.sh \
    outputs/eval/bert_romaji_results.csv \
    outputs/eval/mdeberta_romaji_results.csv \
    outputs/eval/byt5_romaji_results.csv

# All models using wildcard
./scripts/analyze_errors.sh outputs/eval/*_romaji_results.csv

# Default (no arguments) - analyzes BERT and mDeBERTa
./scripts/analyze_errors.sh
```

### Direct Python Usage

```bash
# Single model
python compute_error_taxonomy.py outputs/eval/bert_romaji_results.csv

# Multiple models with comparison
python compute_error_taxonomy.py \
    outputs/eval/bert_romaji_results.csv \
    outputs/eval/mdeberta_romaji_results.csv \
    outputs/eval/byt5_romaji_results.csv \
    --output outputs/eval/error_taxonomy_comparison.json
```

### Output Files

- **Terminal output**: Detailed analysis with interpretations
- **JSON summary**: `outputs/eval/error_taxonomy_comparison.json`
  ```json
  {
    "models": [
      {
        "model_name": "bert_romaji_results",
        "total": 743,
        "type_a": 618,
        "type_b": 50,
        "type_c": 75,
        "bc_ratio": 0.6667,
        "accuracy": 0.8318
      }
    ]
  }
  ```

### Interpreting the Ratio

- **Ratio > 1.0**: Conservative model (misses more toxic than over-flags)
  - For romaji: May indicate tokenization issues preventing detection
  
- **Ratio < 1.0**: Aggressive model (over-flags more than misses)
  - May indicate overfitting or poor generalization
  
- **Ratio ≈ 1.0**: Balanced error distribution

---

## Complete Workflow Example

Here's a complete end-to-end workflow:

```bash
# Step 1: Train all models (if not already trained)
FULL=1 ./scripts/run_tokenizer_models.sh

# Step 2: Run tokenization diagnostics (Section 7.2.1)
# Replace <run_dir> with your actual training output directories
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name microsoft/mdeberta-v3-base \
  --checkpoint outputs/<run_dir>/mdeberta_best_model.pt \
  --output outputs/eval/diagnostics_mdeberta_romaji.json

python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name tohoku-nlp/bert-base-japanese-v3 \
  --checkpoint outputs/training_romaji/bert_data_romaji/bert_romajitrained_best_model.pt \
  --output outputs/eval/diagnostics_bert_japanese_romaji.json

# Step 3: Generate inference results
./scripts/run_taxonomy_analysis_tokenizer_model.sh

# Or run individually:
# python src/inference.py --model <model.pt> --output <output.csv>

# Step 4: Compute error taxonomy statistics (Section 7.2.3)
./scripts/analyze_errors.sh \
    outputs/eval/bert_romaji_results.csv \
    outputs/eval/mdeberta_romaji_results.csv

# Step 5: Review results
cat outputs/eval/error_taxonomy_comparison.json
```

---

## File Naming Conventions

### Diagnostic Files
Format: `diagnostics_<model>_<native|romaji>.json`

Examples:
- `diagnostics_mdeberta_native.json`
- `diagnostics_mdeberta_romaji.json`
- `diagnostics_bert_japanese_native.json`
- `diagnostics_bert_japanese_romaji.json`

### Inference Results
Format: `<model>_romaji_results.csv`

Examples:
- `mdeberta_romaji_results.csv`
- `bert_romaji_results.csv`
- `byt5_romaji_results.csv`

### Error Taxonomy
- Single model: `error_taxonomy_summary.json`
- Multiple models: `error_taxonomy_comparison.json`

---

## Troubleshooting

### Model Checkpoint Not Found
Verify the checkpoint exists:
```bash
ls -lh outputs/<run_dir>/
```
Model checkpoints are generated during training and are not under version control. Locate your actual run directory.

### Data File Not Found
Ensure the paired dataset exists:
```bash
ls -lh data/processed/paired_native_romaji_llmjp_binary.csv
```

### CUDA Out of Memory
The scripts process examples one at a time to minimize memory. If you still encounter OOM:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### Script Permission Denied
Make scripts executable:
```bash
chmod +x scripts/*.sh
chmod +x *.sh
chmod +x compute_error_taxonomy.py
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Train all models | `FULL=1 ./scripts/run_tokenizer_models.sh` |
| Diagnostics | `python3 scripts/tokenization_diagnostics.py --csv <data.csv> --model-name <hf-model> --checkpoint outputs/<run_dir>/<model>.pt --output diagnostics_<name>.json` |
| Inference (single model) | `python src/inference.py --model <model.pt> --output <output.csv>` |
| Batch inference | `./scripts/run_taxonomy_analysis_tokenizer_model.sh` |
| Error taxonomy | `./scripts/analyze_errors.sh outputs/eval/*_romaji_results.csv` |
| View summary | `cat outputs/eval/error_taxonomy_comparison.json` |
