# Training Improvements for Class Imbalance

## Problem Identified

Your previous training run showed the model predicting **only one class** (Non-Toxic):
- **Confusion Matrix**: `[[4 0], [8 0]]` - All 8 toxic samples were misclassified as non-toxic
- **Test Accuracy**: 33.3% (only getting non-toxic samples correct)
- **Root Causes**:
  1. Severe class imbalance (32 non-toxic vs 18 toxic in train, 4 vs 8 in test)
  2. Insufficient training (50 samples, 1 epoch)
  3. No class weighting in loss function

## Changes Made

### 1. **Added Class Weighting to Loss Function**
```python
# In utils.py - SimpleTrainer.__init__
if class_weights is not None:
    class_weights = torch.FloatTensor(class_weights).to(device)
    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Why**: Class weights penalize misclassifications of minority class (toxic) more heavily, forcing the model to learn both classes instead of defaulting to the majority class.

**How it works**: 
- Non-toxic class (60% of data) → weight ≈ 0.83
- Toxic class (40% of data) → weight ≈ 1.25
- Model pays more attention to toxic samples during training

### 2. **Improved Quick Test Mode Settings**
```python
# Changed from:
args.sample_size = 50    → 200
args.epochs = 1          → 3
```

**Why**: 50 samples and 1 epoch is insufficient for a transformer model to learn meaningful patterns. 200 samples with 3 epochs provides better learning while still being fast for testing.

### 3. **Automatic Class Weight Computation**
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
```

**Why**: Automatically calculates optimal weights based on class distribution, adapting to any dataset.

## Expected Improvements

After these changes, you should see:
1. ✅ **Non-zero predictions for toxic class** in confusion matrix
2. ✅ **Improved recall for toxic class** (currently 0%)
3. ✅ **More balanced F1-scores** between classes
4. ✅ **Better overall test accuracy** (>50% expected)

## Recommended Next Steps

### For Better Production Training:

1. **Increase Training Data**
   ```bash
   # Remove --quick-test for full dataset
   python3 src/train.py --model-type byt5 --epochs 5 \
       --data-path ./data/processed/paired_native_romaji_llmjp_binary.csv
   ```

2. **Add Learning Rate Scheduling**
   - Warmup for first 10% of training
   - Linear decay for remaining steps
   - Prevents overshooting optimal weights

3. **Implement Early Stopping**
   - Monitor validation F1-score (not just accuracy)
   - Stop if no improvement for 3 epochs
   - Prevents overfitting

4. **Add Gradient Clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
   - Prevents exploding gradients
   - Especially important for T5/ByT5 models

5. **Use Stratified K-Fold Cross-Validation**
   - Instead of single train/test split
   - More robust performance estimates
   - Better for imbalanced datasets

6. **Monitor Additional Metrics**
   - **Precision/Recall for toxic class** (more important than accuracy)
   - **F1-score** (harmonic mean of precision/recall)
   - **ROC-AUC** (robust to class imbalance)
   - **Matthews Correlation Coefficient** (good for imbalanced binary classification)

### For Evaluation:

1. **Check Class-Specific Performance**
   ```python
   # Focus on toxic class (minority/important class)
   - Precision: Of samples predicted toxic, how many are actually toxic?
   - Recall: Of actual toxic samples, how many did we catch?
   - F1-score: Balance between precision and recall
   ```

2. **Analyze Errors**
   - Which toxic samples are hardest to detect?
   - Are there patterns in false negatives?
   - Does romaji vs native Japanese matter?

3. **Compare Romaji vs Native**
   ```bash
   # Test with romaji
   python3 src/train.py --model-type byt5 --use-romaji \
       --data-path ./data/processed/paired_native_romaji_llmjp_binary.csv
   ```

## Sample Commands

### Quick Test (Fast Validation)
```bash
python3 src/train.py --model-type byt5 --quick-test \
    --data-path ./data/processed/paired_native_romaji_llmjp_binary.csv
```

### Medium Test (Balanced Speed/Quality)
```bash
python3 src/train.py --model-type byt5 --sample-size 500 --epochs 5 \
    --data-path ./data/processed/paired_native_romaji_llmjp_binary.csv
```

### Full Training (Best Performance)
```bash
python3 src/train.py --model-type byt5 --epochs 10 --batch-size 16 \
    --data-path ./data/processed/paired_native_romaji_llmjp_binary.csv
```

### Compare Models
```bash
# ByT5 (byte-level, no tokenizer)
python3 src/train.py --model-type byt5 --epochs 5

# BERT Japanese (word-piece tokenizer)
python3 src/train.py --model-type bert-japanese --epochs 5

# mDeBERTa (multilingual)
python3 src/train.py --model-type mdeberta --epochs 5
```

## Understanding the Output

### Good Signs:
- ✅ Confusion matrix shows non-zero values in all cells
- ✅ Toxic recall > 0.5
- ✅ F1-scores for both classes > 0.5
- ✅ Validation accuracy improving across epochs

### Warning Signs:
- ⚠️ Model only predicting one class
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ Large gap between train and val accuracy
- ⚠️ Toxic recall = 0 (model ignoring minority class)

## Additional Resources

- **Class Imbalance Techniques**: SMOTE, focal loss, cost-sensitive learning
- **ByT5 Paper**: "ByT5: Towards a token-free future with pre-trained byte-to-byte models"
- **Evaluation Metrics**: Beyond accuracy for imbalanced datasets
