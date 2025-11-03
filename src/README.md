# BERT Toxicity Classification

This is a focused implementation for BERT model training and inference on Japanese toxicity classification. It works directly with the existing processed data for quick verification and experimentation.

## Files

- `utils.py` - Core model and utilities (SimpleBertClassifier, SimpleTrainer, load_data, predict_text)
- `train.py` - Training script with support for sample size limiting and quick testing
- `inference.py` - Inference and evaluation script with interactive mode
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

Quick verification (recommended first):
```bash
python3 train.py --quick-test
```

Basic training (native Japanese text):
```bash
python3 train.py
```

Train with limited samples for faster experimentation:
```bash
python3 train.py --sample-size 100
python3 train.py --sample-size 200 --epochs 2
```

Train with romanized text:
```bash
python3 train.py --use-romaji
```

Custom training parameters:
```bash
python3 train.py --epochs 5 --batch-size 32 --learning-rate 3e-5
```

### 5. Run Inference

Test single text:
```bash
python3 inference.py --model outputs/best_model.pt --text "ありがとう"
```

Interactive mode (recommended for testing):
```bash
python3 inference.py --model outputs/best_model.pt --interactive
```

Evaluate on test data:
```bash
python3 inference.py --model outputs/best_model.pt --evaluate
```

Batch inference from file:
```bash
python3 inference.py --model outputs/best_model.pt --texts-file sample_texts.txt
```

## Model Architecture

- Base model: `google-bert/bert-base-multilingual-cased`
- Binary classification (Non-Toxic vs Toxic)
- Uses [CLS] token representation with dropout and linear classifier
- Supports both native Japanese and romanized text

## Training Details

- **Optimizer**: AdamW with learning rate 2e-5 (default)
- **Loss**: CrossEntropyLoss with optional class weights
- **Data Split**: 80/20 train/test split with stratification
- **Model Saving**: Automatic best model saving based on validation accuracy
- **Progress**: Real-time progress tracking with tqdm
- **Quick Testing**: `--quick-test` uses 50 samples, 1 epoch for rapid verification
- **Flexible Sampling**: `--sample-size N` to limit training data for experimentation

## Output

Training creates:
- `outputs/best_model.pt` - Saved model checkpoint (includes model state, tokenizer info, config)
- `outputs/results.json` - Training metrics and evaluation results
- `outputs/config.json` - Training configuration
- `outputs/training.log` - Detailed training logs

Model checkpoint contains:
- `model_state_dict` - PyTorch model weights
- `tokenizer_name` - HuggingFace tokenizer identifier
- `use_romaji` - Whether model was trained on romanized text
- `config` - All training parameters
- `val_acc` - Best validation accuracy achieved

## Command Line Arguments

### Training Arguments
- `--quick-test` - Quick verification: 50 samples, 1 epoch
- `--sample-size N` - Use only N training samples
- `--use-romaji` - Use romanized text instead of native Japanese
- `--epochs N` - Number of training epochs (default: 3)
- `--batch-size N` - Batch size (default: 16)
- `--learning-rate RATE` - Learning rate (default: 2e-5)
- `--data-path PATH` - Path to CSV data file
- `--output-dir DIR` - Output directory (default: outputs)

### Inference Arguments
- `--model PATH` - Path to saved model (required)
- `--text "TEXT"` - Single text to classify
- `--interactive` - Interactive mode
- `--evaluate` - Evaluate on test data
- `--texts-file PATH` - File with texts to classify (one per line)

## Example Usage in Code

```python
from utils import SimpleBertClassifier, predict_text
from transformers import AutoTokenizer
import torch

# Load model with proper error handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('outputs/best_model.pt', map_location=device, weights_only=False)

tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
model = SimpleBertClassifier(checkpoint['tokenizer_name'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Predict single text
result = predict_text(model, tokenizer, "バカ野郎", device)
print(f"Text: {result['text']}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Toxic Probability: {result['toxic_probability']:.3f}")
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
- **Quick test**: 70-80% (limited by small data)
- **Small dataset**: 80-85%
- **Full dataset**: 85-90%

**Note**: If you see 90%+ accuracy but 0% precision/recall on toxic class, your model is likely predicting only non-toxic. This is common with imbalanced small datasets.

## Troubleshooting

### Model Always Predicts Non-Toxic
```bash
# Use more training data
python3 train.py --sample-size 200 --epochs 5

# Check data distribution
python3 explore_data.py
```

### Low Accuracy on Toxic Class
```bash
# Try different learning rate
python3 train.py --learning-rate 1e-5 --epochs 10

# Use full dataset
python3 train.py --epochs 5
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