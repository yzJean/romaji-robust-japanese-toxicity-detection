# Quick Reference Card - Model Evaluation

## 🎯 Common Commands

### 1. Train All Models
```bash
DETERMINISTIC=1 FULL=1 bash scripts/run_tokenizer_models.sh
```

### 2. Tokenization Diagnostics
Note that checkpoint models should be ready for diagnostics.

### Command Template

```bash
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name <huggingface-model-name> \
  --checkpoint <path-to-best-model.pt> \
  --output outputs/eval/diagnostics_<model>_<native|romaji>.json
```

### Examples
```bash
# Tokenization diagnostics (current checkpoint examples)
python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name microsoft/mdeberta-v3-base \
  --checkpoint outputs/training_romaji/mdeberta_romaji_best_model.pt \
  --output outputs/eval/diagnostics_mdeberta_romaji.json

python3 scripts/tokenization_diagnostics.py \
  --csv data/processed/paired_native_romaji_llmjp_binary.csv \
  --model-name tohoku-nlp/bert-base-japanese-v3 \
  --checkpoint outputs/training_romaji/bert_romaji_best_model.pt \
  --output outputs/eval/diagnostics_bert_japanese_romaji.json
```

### 3. Inference (Pick approach)
```bash
# Batch (both BERT + mDeBERTa)
./scripts/run_taxonomy_analysis_tokenizer_model.sh

# Individual (generic)
python src/inference.py --model <model.pt> --output <output.csv>

# Examples: romaji-trained BERT checkpoint — native and romaji evaluations
python3 src/inference.py \
  --model outputs/training_romaji/bert_romaji_best_model.pt \
  --output outputs/eval/bert_romajitrained_nativetest_results.csv --language native

python3 src/inference.py \
  --model outputs/training_romaji/bert_data_romaji/bert_romaji_best_model.pt \
  --output outputs/eval/bert_romajitrained_romajitest_results.csv --language romaji
```

### 4. Error Taxonomy
```bash
# All models
./scripts/analyze_errors.sh outputs/eval/*_romaji_results.csv

# Specific models
./scripts/analyze_errors.sh \
    outputs/eval/bert_romaji_results.csv \
    outputs/eval/mdeberta_romaji_results.csv \
    outputs/eval/byt5_romaji_results.csv
```

## 📁 Key File Locations

### Input Data
- `data/processed/paired_native_romaji_llmjp_binary.csv`

### Model Checkpoints
- `outputs/tokenizer_runs/<run_dir>/<model>_best_model.pt`
  - Example: `outputs/tokenizer_runs/run_2_model_type_mdeberta_data_path_..._use_romaji/mdeberta_best_model.pt`

### Outputs
- Diagnostics: `outputs/eval/diagnostics_*.json`
- Inference: `outputs/eval/*_romaji_results.csv`
- Taxonomy: `outputs/eval/error_taxonomy_comparison.json`

## 🔧 Troubleshooting

```bash
# Make scripts executable
chmod +x scripts/*.sh *.sh compute_error_taxonomy.py

# Force CPU (if CUDA OOM)
export CUDA_VISIBLE_DEVICES=""

# Check files exist
ls -lh outputs/training_romaji/*/
ls -lh data/processed/
```

## 📊 Output Files

| File | Purpose |
|------|---------|
| `diagnostics_<model>_<script>.json` | Tokenization metrics (7.2.1) |
| `<model>_romaji_results.csv` | Inference predictions |
| `error_taxonomy_comparison.json` | Type B/C ratio, statistics (7.2.3) |

## 📖 Full Documentation

See **[EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md)** for complete details.
