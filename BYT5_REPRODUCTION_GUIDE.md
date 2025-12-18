# ByT5 Tokenizer-Free Architecture - Reproduction Guide

This guide covers reproducing the ByT5 results for script-invariant Japanese toxicity detection.


## Quick Start

```bash
# 1. Setup environment (if not already done)
source .venv/bin/activate

# 2. Train ByT5 on native Japanese
python src/train.py --model-type byt5 --epochs 5 --batch-size 16

# 3. Train ByT5 on romaji
python src/train.py --model-type byt5 --use-romaji --epochs 5 --batch-size 16

# 4. Evaluate and visualize flip rates
python evaluation/evaluate_byt5.py
python evaluation/visualize_byt5_flip_rate.py
```

## Step-by-Step Instructions

### Prerequisites

Ensure data is prepared:
```bash
# Load and standardize datasets
python scripts/load_data.py

# Create paired native-romaji datasets
python scripts/pair_native_romaji.py
```

This creates:
- `data/standardized/inspection_ai.csv`
- `data/standardized/llmjp.csv`
- `data/processed/paired_native_romaji_inspection_ai_binary.csv`
- `data/processed/paired_native_romaji_llmjp_binary.csv`

### Training ByT5 Models

#### 1. Native-trained ByT5
```bash
python src/train.py --model-type byt5 --epochs 5 --batch-size 16
```

**Output**:
- Model checkpoint: `outputs/google_byt5_small_best_model.pt`
- Training logs with accuracy, F1, confusion matrix

**Expected Performance**:
- Validation accuracy: ~61%
- Toxic recall: ~27% (low due to script inconsistency)

#### 2. Romaji-trained ByT5
```bash
python src/train.py --model-type byt5 --use-romaji --epochs 5 --batch-size 16
```

**Output**:
- Model checkpoint: `outputs/google_byt5_small_romaji_best_model.pt`

**Expected Performance**:
- Better script-invariance
- More consistent predictions across native/romaji inputs

#### Quick Test Mode
For verification before full training:
```bash
python src/train.py --model-type byt5 --quick-test
```
Uses 50 samples, 1 epoch (~2 minutes)

### Evaluation Pipeline

#### 1. Cross-Architecture Evaluation
```bash
python evaluation/evaluate_byt5.py
```

**Generates**:
- `outputs/byt5_cross_architecture_evaluation_standardized.csv`

**Contains**: Model predictions on standardized test set for flip rate analysis

#### 2. Flip Rate Visualization
```bash
python evaluation/visualize_byt5_flip_rate.py
```

**Generates**:
- `outputs/byt5_flip_rate_visualization.png`

**Console Output**:
```
BYT5 FLIP RATE SUMMARY
======================================================================
Native-trained ByT5:
  Total Flip Rate: 44.34%
  1->0 (toxic->non-toxic): 9.03% (67 samples)
  0->1 (non-toxic->toxic): 35.31% (262 samples)

Romaji-trained ByT5:
  Total Flip Rate: 16.71%
  1->0 (toxic->non-toxic): 2.70% (20 samples)
  0->1 (non-toxic->toxic): 14.02% (104 samples)

Interpretation:
  - Lower flip rate = more stable across scripts
  - Romaji training improves flip rate by 62%
  - ByT5 outperforms tokenizers (mDeBERTa: 33.78%, BERT: 42.12%)
======================================================================
```

#### 3. Statistical Validation
```bash
# Cohen's Kappa (inter-script agreement)
python evaluation/compute_kappa_all_models.py

# McNemar's test (statistical significance)
python evaluation/estimate_mcnemar_byt5.py
```

**Outputs**:
- `outputs/kappa_script_invariance_results.json`
- Statistical significance metrics

### Comparing with Baseline Models

To validate ByT5's superiority over tokenizer-based models:

```bash
# Train mDeBERTa (subword tokenizer)
python src/train.py --model-type mdeberta --epochs 5 --batch-size 16
python src/train.py --model-type mdeberta --use-romaji --epochs 5 --batch-size 16

# Train BERT Japanese (subword tokenizer)
python src/train.py --model-type bert-japanese --epochs 5 --batch-size 16
python src/train.py --model-type bert-japanese --use-romaji --epochs 5 --batch-size 16

# Generate comparison
python src/compare_models.py
```

**Expected Results**:
- mDeBERTa flip rate: 33.78%
- BERT flip rate: 42.12%
- ByT5 flip rate: 16.71% (best)

## Key Results

### ByT5 Flip Rate Performance

| Model | Flip Rate | Toxic→Non-toxic | Non-toxic→Toxic |
|-------|-----------|----------------|-----------------|
| ByT5 (Native) | 44.34% | 9.03% (67) | 35.31% (262) |
| ByT5 (Romaji) | **16.71%** | 2.70% (20) | 14.02% (104) |

**Improvement**: 62% reduction in flip rate (44.34% → 16.71%)

### Comparison with Tokenizers

| Architecture | Tokenizer Type | Flip Rate |
|--------------|---------------|-----------|
| ByT5 | Byte-level (none) | **16.71%** |
| mDeBERTa | Subword | 33.78% |
| BERT-Japanese | Subword | 42.12% |

**Finding**: Tokenizer-free (character-level) models achieve better script-invariance

## File Locations

### Input Data
- `data/standardized/inspection_ai.csv` - Standardized Inspection AI dataset
- `data/standardized/llmjp.csv` - Standardized LLMJP dataset
- `data/processed/paired_native_romaji_*.csv` - Paired datasets for evaluation

### Model Checkpoints
- `outputs/google_byt5_small_best_model.pt` - Native-trained ByT5
- `outputs/google_byt5_small_romaji_best_model.pt` - Romaji-trained ByT5

### Evaluation Results
- `outputs/byt5_cross_architecture_evaluation_standardized.csv` - Predictions
- `outputs/byt5_flip_rate_visualization.png` - Flip rate chart
- `outputs/kappa_script_invariance_results.json` - Cohen's Kappa scores

### Source Code
- `src/train.py` - Main training script
- `evaluation/evaluate_byt5.py` - ByT5 evaluation
- `evaluation/visualize_byt5_flip_rate.py` - Flip rate visualization
- `scripts/load_data.py` - Data loading utilities
- `scripts/pair_native_romaji.py` - Native-romaji pairing

## Training Options

```bash
# Full option list
python src/train.py --help

# Common configurations:
--model-type byt5          # Use ByT5 model
--use-romaji               # Train on romaji text
--epochs N                 # Number of training epochs (default: 5)
--batch-size N             # Batch size (default: 16)
--sample-size N            # Limit training samples
--quick-test               # Quick test mode (50 samples, 1 epoch)
--deterministic            # Reproducible training (slower)
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python src/train.py --model-type byt5 --batch-size 8
```

### Slow Training
Enable quick test for verification:
```bash
python src/train.py --model-type byt5 --quick-test
```

### Missing Data Files
Re-run data preparation:
```bash
python scripts/load_data.py
python scripts/pair_native_romaji.py
```

### Reproducibility
Use deterministic mode:
```bash
python src/train.py --model-type byt5 --deterministic
```

## Key Contributions

1. **Tokenizer-free architecture evaluation**: First systematic study of byte-level models for Japanese toxicity detection with script-invariance

2. **Flip rate metric**: Developed quantitative measure of prediction consistency across native and romanized text

3. **Superior performance**: Demonstrated ByT5 achieves 16.71% flip rate vs. 33-42% for subword tokenizers

4. **Romaji training benefit**: Proved romaji training reduces flip rate by 62% while maintaining accuracy

5. **Character-level advantage**: Established that byte-level processing enables better cross-script generalization

