# Multi-Model Toxicity Classification

This is a focused implementation for transformer-based model training and inference on Japanese toxicity classification. It supports both BERT and XLM-RoBERTa models and works directly with the existing processed data for quick verification and experimentation.

## Files

- `utils.py` - Core model and utilities (SimpleBertClassifier, SimpleTrainer, load_data, predict_text)
- `train.py` - Training script with multi-model support, sample size limiting and quick testing
- `inference.py` - Inference and evaluation script with interactive mode
- `compare_models.py` - Model comparison utility for BERT vs XLM-RoBERTa
- `explore_data.py` - Data exploration and analysis utility

## Quick Start

### 1. Install Dependencies

```bash
pip install torch scikit-learn matplotlib seaborn
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have the processed data file:
```
data/processed/paired_inspection_ai_binary.csv
```

This should contain columns: `id`, `text_native`, `text_romaji`, `label_int_coarse`, `label_text_fine`, `source`

### 3. Explore Your Data (Optional)

Check data size and distribution:
```bash
cd src
python3 explore_data.py
```

### 4. Train Model

Quick verification with BERT (recommended first):
```bash
python3 train.py --quick-test
```

Train BERT model:
```bash
python3 train.py --model-type bert --epochs 10
```

Train XLM-RoBERTa model:
```bash
python3 train.py --model-type xlm-roberta --epochs 10
```

Quick test with XLM-RoBERTa:
```bash
python3 train.py --xlm-roberta --quick-test
```

Train with limited samples for faster experimentation:
```bash
python3 train.py --model-type bert --sample-size 100
python3 train.py --model-type xlm-roberta --sample-size 200 --epochs 2
```

Train with romanized text:
```bash
python3 train.py --model-type bert --use-romaji
python3 train.py --model-type xlm-roberta --use-romaji
```

Custom training parameters:
```bash
python3 train.py --model-type bert --epochs 5 --batch-size 32 --learning-rate 3e-5
python3 train.py --model-type xlm-roberta --epochs 5 --batch-size 32 --learning-rate 2e-5
```

Compare both models:
```bash
python3 compare_models.py --quick-test
python3 compare_models.py --epochs 5 --sample-size 500
```

### 5. Run Inference

Test single text with BERT model:
```bash
python3 inference.py --model outputs/google_bert_bert_base_multilingual_cased_best_model.pt --text "ありがとう"
```

Test single text with XLM-RoBERTa model:
```bash
python3 inference.py --model outputs/FacebookAI_xlm_roberta_base_best_model.pt --text "ありがとう"
```

Interactive mode (recommended for testing):
```bash
python3 inference.py --model outputs/google_bert_bert_base_multilingual_cased_best_model.pt --interactive
python3 inference.py --model outputs/FacebookAI_xlm_roberta_base_best_model.pt --interactive
```

Evaluate on test data:
```bash
python3 inference.py --model outputs/google_bert_bert_base_multilingual_cased_best_model.pt --evaluate
python3 inference.py --model outputs/FacebookAI_xlm_roberta_base_best_model.pt --evaluate
```

Batch inference from file:
```bash
python3 inference.py --model outputs/google_bert_bert_base_multilingual_cased_best_model.pt --texts-file sample_texts.txt
python3 inference.py --model outputs/FacebookAI_xlm_roberta_base_best_model.pt --texts-file sample_texts.txt
```

## Supported Models

| Model | Identifier | Usage |
|-------|------------|-------|
| **BERT Multilingual** | `google-bert/bert-base-multilingual-cased` | `--model-type bert` (default) |
| **XLM-RoBERTa** | `FacebookAI/xlm-roberta-base` | `--model-type xlm-roberta` or `--xlm-roberta` |

## Model Architecture

- **Architecture**: Transformer encoder with classification head
- **Classification**: Binary classification (Non-Toxic vs Toxic)
- **Token Representation**: Uses [CLS] token representation with dropout and linear classifier
- **Text Support**: Supports both native Japanese and romanized text
- **Model Selection**: Automatic model loading based on HuggingFace identifiers

## Training Details

- **Optimizer**: AdamW with learning rate 2e-5 (default)
- **Loss**: CrossEntropyLoss with optional class weights
- **Data Split**: 80/20 train/test split with stratification
- **Model Saving**: Automatic best model saving based on validation accuracy
- **Progress**: Real-time progress tracking with tqdm
- **Quick Testing**: `--quick-test` uses 50 samples, 1 epoch for rapid verification
- **Flexible Sampling**: `--sample-size N` to limit training data for experimentation

## Output

Training creates model-specific files:

**For BERT model:**
- `outputs/google_bert_bert_base_multilingual_cased_best_model.pt` - BERT model checkpoint
- `outputs/google_bert_bert_base_multilingual_cased_results.json` - BERT training results
- `outputs/google_bert_bert_base_multilingual_cased_config.json` - BERT training configuration

**For XLM-RoBERTa model:**
- `outputs/FacebookAI_xlm_roberta_base_best_model.pt` - XLM-RoBERTa model checkpoint
- `outputs/FacebookAI_xlm_roberta_base_results.json` - XLM-RoBERTa training results
- `outputs/FacebookAI_xlm_roberta_base_config.json` - XLM-RoBERTa training configuration

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
- `--model-type {bert,xlm-roberta}` - Choose model type (bert is default)
- `--xlm-roberta` - Shortcut to use XLM-RoBERTa model
- `--quick-test` - Quick verification: 50 samples, 1 epoch
- `--sample-size N` - Use only N training samples
- `--use-romaji` - Use romanized text instead of native Japanese
- `--epochs N` - Number of training epochs (default: 3)
- `--batch-size N` - Batch size (default: 16)
- `--learning-rate RATE` - Learning rate (default: 2e-5)
- `--data-path PATH` - Path to CSV data file
- `--output-dir DIR` - Output directory (default: outputs)

### Model Comparison Arguments
- `--quick-test` - Quick comparison: 100 samples, 2 epochs per model
- `--sample-size N` - Use N training samples for each model
- `--use-romaji` - Compare models on romanized text
- `--epochs N` - Number of epochs for each model (default: 3)
- `--output-dir DIR` - Directory to save comparison results

### Inference Arguments
- `--model PATH` - Path to saved model checkpoint (required)
- `--text "TEXT"` - Single text to classify
- `--interactive` - Interactive mode for testing multiple texts
- `--evaluate` - Evaluate model performance on test data
- `--texts-file PATH` - File with texts to classify (one per line)
- `--use-romaji` - Use romanized text (must match training setting)

## Model Comparison

Compare BERT and XLM-RoBERTa performance:

```bash
# Quick comparison (recommended first)
python3 compare_models.py --quick-test

# Full comparison
python3 compare_models.py --epochs 5

# Compare with specific sample size
python3 compare_models.py --sample-size 300 --epochs 3

# Compare on romanized text
python3 compare_models.py --use-romaji --quick-test
```

The comparison script will:
- Train both models with identical settings
- Show side-by-side performance metrics
- Highlight the best performing model
- Save detailed results to `outputs/comparison/model_comparison.json`

## Example Usage in Code

```python
from utils import SimpleBertClassifier, predict_text
from transformers import AutoTokenizer
import torch

# Load BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_checkpoint = torch.load('outputs/google_bert_bert_base_multilingual_cased_best_model.pt',
                            map_location=device, weights_only=False)

bert_tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint['tokenizer_name'])
bert_model = SimpleBertClassifier(bert_checkpoint['tokenizer_name'])
bert_model.load_state_dict(bert_checkpoint['model_state_dict'])
bert_model.to(device)
bert_model.eval()

# Load XLM-RoBERTa model
xlm_checkpoint = torch.load('outputs/FacebookAI_xlm_roberta_base_best_model.pt',
                           map_location=device, weights_only=False)

xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_checkpoint['tokenizer_name'])
xlm_model = SimpleBertClassifier(xlm_checkpoint['tokenizer_name'])
xlm_model.load_state_dict(xlm_checkpoint['model_state_dict'])
xlm_model.to(device)
xlm_model.eval()

# Compare predictions
text = "バカ野郎"
bert_result = predict_text(bert_model, bert_tokenizer, text, device)
xlm_result = predict_text(xlm_model, xlm_tokenizer, text, device)

print(f"Text: {text}")
print(f"BERT: {bert_result['prediction']} ({bert_result['confidence']:.3f})")
print(f"XLM-RoBERTa: {xlm_result['prediction']} ({xlm_result['confidence']:.3f})")
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
- ~2-4GB GPU memory for batch size 16
- ~1-2GB RAM for dataset loading

### Expected Accuracy

**BERT Multilingual:**
- **Quick test**: 70-80% (limited by small data)
- **Small dataset**: 80-85%
- **Full dataset**: 85-90%

**XLM-RoBERTa:**
- **Quick test**: 70-85% (often slightly better than BERT)
- **Small dataset**: 82-87%
- **Full dataset**: 87-92%

**Note**: If you see 90%+ accuracy but 0% precision/recall on toxic class, your model is likely predicting only non-toxic. This is common with imbalanced small datasets. XLM-RoBERTa typically performs better on multilingual tasks.

## Troubleshooting

### Model Always Predicts Non-Toxic
```bash
# Use more training data with BERT
python3 train.py --model-type bert --sample-size 200 --epochs 5

# Try XLM-RoBERTa (often better on imbalanced data)
python3 train.py --model-type xlm-roberta --sample-size 200 --epochs 5

# Check data distribution
python3 explore_data.py
```

### Low Accuracy on Toxic Class
```bash
# Try different learning rate with BERT
python3 train.py --model-type bert --learning-rate 1e-5 --epochs 10

# Try XLM-RoBERTa with full dataset
python3 train.py --model-type xlm-roberta --epochs 5

# Compare both models
python3 compare_models.py --epochs 5
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