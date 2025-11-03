# BERT Toxicity Classification

This is a focused implementation for BERT model training and inference on Japanese toxicity classification. It works directly with the existing processed data for quick verification and experimentation.

## Files

- `bert_toxicity.py` - Core model and utilities
- `train.py` - Training script
- `inference.py` - Inference and evaluation script

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

### 3. Train Model

Basic training (native Japanese text):
```bash
cd src
python train.py
```

Train with romanized text:
```bash
python train.py --use-romaji
```

Custom training parameters:
```bash
python train.py --epochs 5 --batch-size 32 --learning-rate 3e-5
```

### 4. Run Inference

Test single text:
```bash
python inference.py --model outputs/best_model.pt --text "ありがとう"
```

Interactive mode:
```bash
python inference.py --model outputs/best_model.pt --interactive
```

Evaluate on test data:
```bash
python inference.py --model outputs/best_model.pt --evaluate
```

## Model Architecture

- Base model: `google-bert/bert-base-multilingual-cased`
- Binary classification (Non-Toxic vs Toxic)
- Uses [CLS] token representation with dropout and linear classifier
- Supports both native Japanese and romanized text

## Training Details

- AdamW optimizer with learning rate 2e-5
- CrossEntropyLoss
- 80/20 train/test split with stratification
- Automatic best model saving based on validation accuracy
- Progress tracking with tqdm

## Output

Training creates:
- `outputs/best_model.pt` - Saved model checkpoint
- `outputs/results.json` - Training metrics and evaluation results
- `outputs/config.json` - Training configuration

## Example Usage in Code

```python
from bert_toxicity import SimpleBertClassifier, predict_text
from transformers import AutoTokenizer
import torch

# Load model
checkpoint = torch.load('outputs/best_model.pt')
tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])

model = SimpleBertClassifier(checkpoint['tokenizer_name'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
result = predict_text(model, tokenizer, "バカ野郎", device)
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
```

## Data Format

The implementation expects CSV data with these columns:
- `text_native`: Original Japanese text
- `text_romaji`: Romanized version (if using --use-romaji)
- `label_int_coarse`: Binary labels (0=Non-Toxic, 1=Toxic)

## Performance

On typical Japanese toxicity data:
- Training time: ~5-10 minutes on GPU for 3 epochs
- Inference speed: ~100-200 samples/second on GPU
- Memory usage: ~2-4GB GPU memory for batch size 16

## Limitations

This is a simplified implementation for quick verification:
- No advanced data augmentation
- Basic train/test split (no validation set)
- Simple evaluation metrics
- No hyperparameter tuning
- No cross-validation

For production use, consider the full implementation with proper data pipeline integration.