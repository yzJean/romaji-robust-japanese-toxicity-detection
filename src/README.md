# Multi-Model Toxicity Classification

This is a focused implementation for transformer-based model training and inference on Japanese toxicity classification. It supports mDeBERTa-v3 and BERT Japanese models and works directly with the existing processed data for quick verification and experimentation.

## Files

- `utils.py` - Core model and utilities (SimpleBertClassifier, SimpleTrainer, load_data, predict_text)
- `train.py` - Training script with multi-model support, sample size limiting, quick testing, and gradient accumulation
- `inference.py` - Inference and evaluation script with interactive mode
- `compare_models.py` - Model comparison utility for mDeBERTa-v3 vs BERT Japanese
- `explore_data.py` - Data exploration and analysis utility

## Quick Start

### 1. Install Dependencies

```bash
pip install torch scikit-learn matplotlib seaborn transformers
```

**Model-specific dependencies:**

For **mDeBERTa** model:
```bash
pip install sentencepiece tiktoken protobuf
```

For **BERT Japanese** model:
```bash
pip install fugashi unidic-lite
```

Or install all requirements:
```bash
pip install -r requirements.txtpip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have the processed data file. The default path is:
```
data/processed/paired_native_romaji_inspection_ai_binary.csv
```

Other supported data files:
```
data/processed/paired_native_romaji_llmjp_binary.csv
```

Expected columns: `text_native`, `text_romaji`, `label_int_coarse`

### 3. Explore Your Data (Optional)

Check data size and distribution:
```bash
cd src
python3 explore_data.py
```

### 4. Train Model

Quick verification with mDeBERTa (default):
```bash
python3 train.py --quick-test
```

Quick test with BERT Japanese:
```bash
python3 train.py --model-type bert-japanese --quick-test
```

Train mDeBERTa model:
```bash
python3 train.py --model-type mdeberta --epochs 10
```

Train BERT Japanese model:
```bash
python3 train.py --model-type bert-japanese --epochs 10
```

**Memory optimization for GPU:**

If you encounter out-of-memory errors, reduce batch size:
```bash
python3 train.py --model-type mdeberta --epochs 10 --batch-size 8
python3 train.py --model-type mdeberta --epochs 10 --batch-size 4
```

Train with limited samples for faster experimentation:
```bash
python3 train.py --model-type mdeberta --sample-size 100
python3 train.py --model-type bert-japanese --sample-size 200 --epochs 2
```

Train with romanized text:
```bash
python3 train.py --model-type mdeberta --use-romaji
python3 train.py --model-type bert-japanese --use-romaji
```

Custom training parameters:
```bash
python3 train.py --model-type mdeberta --epochs 5 --batch-size 32 --learning-rate 3e-5
python3 train.py --model-type bert-japanese --epochs 5 --batch-size 32 --learning-rate 2e-5
```

Compare both models:
```bash
python3 compare_models.py --quick-test
python3 compare_models.py --epochs 5 --sample-size 500
```

Tokenizer vs byte-level comparison (mDeBERTa vs ByT5):
```bash
python3 tokenizer_vs_byt5.py --epochs 3
```

### 5. Run Inference

Test single text with mDeBERTa model:
```bash
python3 src/inference.py --model outputs/microsoft_mdeberta_v3_base_best_model.pt --text "ありがとう"
```

Test single text with BERT Japanese model:
```bash
python3 src/inference.py --model outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt --text "ありがとう"
```

Interactive mode (recommended for testing):
```bash
python3 src/inference.py --model outputs/microsoft_mdeberta_v3_base_best_model.pt --interactive
python3 src/inference.py --model outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt --interactive
```

Evaluate on test data:
```bash
python3 src/inference.py --model outputs/microsoft_mdeberta_v3_base_best_model.pt --evaluate
python3 src/inference.py --model outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt --evaluate
```

Batch inference from file:
```bash
python3 src/inference.py --model outputs/microsoft_mdeberta_v3_base_best_model.pt --texts-file sample_texts.txt
python3 src/inference.py --model outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt --texts-file sample_texts.txt
```

## Supported Models

| Model | Identifier | Usage | Best For |
|-------|------------|-------|----------|
| **mDeBERTa-v3** | `microsoft/mdeberta-v3-base` | `--model-type mdeberta` (default) | Multilingual text, romaji support |
| **BERT Japanese** | `tohoku-nlp/bert-base-japanese-v3` | `--model-type bert-japanese` | Native Japanese text only |

### Model Characteristics

**mDeBERTa-v3:**
- ✅ Supports 100+ languages including Japanese
- ✅ Good performance on romaji (romanized text)
- ✅ No external dependencies beyond Python packages
- ⚠️ Higher memory usage (may need smaller batch sizes)

**BERT Japanese:**
- ✅ Optimized specifically for Japanese language
- ✅ Better tokenization for native Japanese text
- ✅ Trained on Japanese corpus
- ⚠️ Poor performance on romaji (romanized text)
- ⚠️ Use only with `text_native`, not with `--use-romaji`

## Model Architecture

- **Architecture**: Transformer encoder with classification head
- **Classification**: Binary classification (Non-Toxic vs Toxic)
- **Token Representation**: Uses [CLS] token representation with dropout and linear classifier
- **Text Support**: Supports both native Japanese and romanized text
- **Model Selection**: Automatic model loading based on HuggingFace identifiers

## Training Details

- **Optimizer**: AdamW with learning rate 2e-5 (default)
- **Loss**: CrossEntropyLoss
- **Data Split**: 80/20 train/test split with stratification
- **Model Saving**: Automatic best model saving based on validation accuracy
- **Progress**: Real-time progress tracking with tqdm
- **Quick Testing**: `--quick-test` uses 50 samples, 1 epoch, batch size 8 for rapid verification
- **Flexible Sampling**: `--sample-size N` to limit training data for experimentation
- **Gradient Accumulation**: `--gradient-accumulation-steps N` to maintain effective batch size with lower memory usage

## Output

Training creates model-specific files:

**For mDeBERTa model:**
- `outputs/microsoft_mdeberta_v3_base_best_model.pt` - mDeBERTa model checkpoint
- `outputs/microsoft_mdeberta_v3_base_results.json` - mDeBERTa training results
- `outputs/microsoft_mdeberta_v3_base_config.json` - mDeBERTa training configuration

**For BERT Japanese model:**
- `outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt` - BERT Japanese model checkpoint
- `outputs/tohoku_nlp_bert_base_japanese_v3_results.json` - BERT Japanese training results
- `outputs/tohoku_nlp_bert_base_japanese_v3_config.json` - BERT Japanese training configuration

**For model comparison:**
- `outputs/comparison/model_comparison.json` - Side-by-side comparison results

Model checkpoint contains:
- `model_state_dict` - PyTorch model weights
- `tokenizer_name` - HuggingFace tokenizer identifier
- `use_romaji` - Whether model was trained on romanized text
- `config` - All training parameters
- `val_acc` - Best validation accuracy achieved

## Command Line Arguments

### Training Arguments
- `--model-type {mdeberta,bert-japanese}` - Choose model type (mdeberta is default)
- `--quick-test` - Quick verification: 50 samples, 1 epoch, batch size 8
- `--sample-size N` - Use only N training samples
- `--use-romaji` - Use romanized text (recommended only with mdeberta)
- `--epochs N` - Number of training epochs (default: 3)
- `--batch-size N` - Batch size (default: 16)
- `--gradient-accumulation-steps N` - Gradient accumulation steps (default: 1)
- `--learning-rate RATE` - Learning rate (default: 2e-5)
- `--dropout RATE` - Dropout rate (default: 0.1)
- `--max-length N` - Maximum sequence length (default: 512)
- `--test-size FLOAT` - Test set fraction (default: 0.2)
- `--data-path PATH` - Path to CSV data file
- `--output-dir DIR` - Output directory (default: outputs)

### Model Comparison Arguments
- `--quick-test` - Quick comparison: 100 samples, 2 epochs per model
- `--sample-size N` - Use N training samples for each model
- `--use-romaji` - Compare models on romanized text
- `--epochs N` - Number of epochs for each model (default: 3)
- `--batch-size N` - Batch size (default: 16)
- `--learning-rate RATE` - Learning rate (default: 2e-5)
- `--dropout RATE` - Dropout rate (default: 0.1)
- `--max-length N` - Maximum sequence length (default: 512)
- `--test-size FLOAT` - Test set fraction (default: 0.2)
- `--data-path PATH` - Path to CSV data file
- `--output-dir DIR` - Directory to save comparison results (default: outputs/comparison)

### Inference Arguments
- `--model PATH` - Path to saved model checkpoint (required)
- `--text "TEXT"` - Single text to classify
- `--interactive` - Interactive mode for testing multiple texts
- `--evaluate` - Evaluate model performance on test data
- `--texts-file PATH` - File with texts to classify (one per line)
- `--data-path PATH` - Path to data file for evaluation (default: data/processed/paired_inspection_ai_binary.csv)
- `--output PATH` - Output file to save results (JSON format)

## Model Comparison

Compare mDeBERTa and BERT Japanese performance:

```bash
# Quick comparison (recommended first)
python3 compare_models.py --quick-test
python3 compare_models.py --quick-test --data-path ../data/processed/paired_inspection_ai_binary.csv

# Full comparison
python3 compare_models.py --epochs 5

# Compare with specific sample size
python3 compare_models.py --sample-size 300 --epochs 3

# Compare on romanized text (only mDeBERTa will perform well)
python3 compare_models.py --use-romaji --quick-test
```

The comparison script will:
- Train both models with identical settings
- Show side-by-side performance metrics
- Highlight the best performing model
- Save detailed results to `outputs/comparison/model_comparison.json`

**Note**: When comparing with `--use-romaji`, expect mDeBERTa to significantly outperform BERT Japanese, as BERT Japanese is not designed for romanized text.

## Example Usage in Code

```python
from utils import SimpleBertClassifier, predict_text
from transformers import AutoTokenizer
import torch

# Load mDeBERTa model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mdeberta_checkpoint = torch.load('outputs/microsoft_mdeberta_v3_base_best_model.pt',
                                 map_location=device, weights_only=False)

mdeberta_tokenizer = AutoTokenizer.from_pretrained(mdeberta_checkpoint['tokenizer_name'])
mdeberta_model = SimpleBertClassifier(mdeberta_checkpoint['tokenizer_name'])
mdeberta_model.load_state_dict(mdeberta_checkpoint['model_state_dict'])
mdeberta_model.to(device)
mdeberta_model.eval()

# Load BERT Japanese model
bert_jp_checkpoint = torch.load('outputs/tohoku_nlp_bert_base_japanese_v3_best_model.pt',
                                map_location=device, weights_only=False)

bert_jp_tokenizer = AutoTokenizer.from_pretrained(bert_jp_checkpoint['tokenizer_name'])
bert_jp_model = SimpleBertClassifier(bert_jp_checkpoint['tokenizer_name'])
bert_jp_model.load_state_dict(bert_jp_checkpoint['model_state_dict'])
bert_jp_model.to(device)
bert_jp_model.eval()

# Compare predictions on native Japanese text
text = "バカ野郎"
mdeberta_result = predict_text(mdeberta_model, mdeberta_tokenizer, text, device)
bert_jp_result = predict_text(bert_jp_model, bert_jp_tokenizer, text, device)

print(f"Text: {text}")
print(f"mDeBERTa: {mdeberta_result['prediction']} ({mdeberta_result['confidence']:.3f})")
print(f"BERT Japanese: {bert_jp_result['prediction']} ({bert_jp_result['confidence']:.3f})")
```

## Data Format

The implementation expects CSV data with these columns:
- `text_native`: Original Japanese text
- `text_romaji`: Romanized version (if using --use-romaji)
- `label_int_coarse`: Binary labels (0=Non-Toxic, 1=Toxic)

## Performance & Expected Results

### Training Times
- **Quick test** (50 samples): ~1-2 minutes
- **Small dataset** (100-200 samples): ~2-5 minutes
- **Full dataset** (~300 samples): ~10-15 minutes on GPU

### Inference Speed
- ~100-200 samples/second on GPU
- ~10-20 samples/second on CPU

### Memory Usage
- **mDeBERTa-v3**: ~4-8GB GPU memory for batch size 16 (higher than BERT Japanese)
- **BERT Japanese**: ~2-4GB GPU memory for batch size 16
- **RAM**: ~1-2GB for dataset loading
- **Tip**: Use `--batch-size 4 --gradient-accumulation-steps 8` for 32GB GPU to maintain effective batch size of 32

### Expected Accuracy

**mDeBERTa-v3:**
- **Quick test**: 70-85% (limited by small data)
- **Small dataset**: 82-88%
- **Full dataset**: 87-92%
- **With romaji**: Similar performance to native text

**BERT Japanese:**
- **Quick test**: 70-80% (limited by small data)
- **Small dataset**: 80-86%
- **Full dataset**: 85-90%
- **With romaji**: Poor performance (not recommended)

**Note**: If you see 90%+ accuracy but 0% precision/recall on toxic class, your model is likely predicting only non-toxic. This is common with imbalanced small datasets. mDeBERTa typically performs better on multilingual and romanized text.

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
python3 train.py --model-type mdeberta --batch-size 8

# Use gradient accumulation (maintains effective batch size)
python3 train.py --model-type mdeberta --batch-size 4 --gradient-accumulation-steps 8

# Try with even smaller batch
python3 train.py --model-type mdeberta --batch-size 2 --gradient-accumulation-steps 16
```

### Model Always Predicts Non-Toxic
```bash
# Use more training data with mDeBERTa
python3 train.py --model-type mdeberta --sample-size 200 --epochs 5

# Try BERT Japanese (better for imbalanced native Japanese data)
python3 train.py --model-type bert-japanese --sample-size 200 --epochs 5

# Check data distribution
python3 explore_data.py
```

### Low Accuracy on Toxic Class
```bash
# Try different learning rate with mDeBERTa
python3 train.py --model-type mdeberta --learning-rate 1e-5 --epochs 10

# Try BERT Japanese with full dataset
python3 train.py --model-type bert-japanese --epochs 5

# Compare both models
python3 compare_models.py --epochs 5
```

### BERT Japanese Poor Performance
```bash
# Make sure NOT using romaji
python3 train.py --model-type bert-japanese  # uses text_native by default

# BERT Japanese does not work well with romaji - use mDeBERTa instead
python3 train.py --model-type mdeberta --use-romaji
```

### Missing Dependencies
```bash
# For mDeBERTa
pip install sentencepiece tiktoken protobuf

# For BERT Japanese
pip install fugashi unidic-lite

# Install all
pip install -r requirements.txt
```

### PyTorch Loading Errors
The code handles PyTorch version compatibility automatically. Models are saved with `weights_only=False` support.

## Implementation Notes

This implementation prioritizes:
- **Simplicity**: Easy to understand and modify
- **Speed**: Quick experimentation and verification
- **Flexibility**: Command-line arguments for all parameters
- **Robustness**: Error handling for common issues

For production use with larger datasets and more complex requirements, consider extending with:
- Advanced data augmentation
- Cross-validation
- Hyperparameter optimization
- Class balancing strategies
- More sophisticated evaluation metrics