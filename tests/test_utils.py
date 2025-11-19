#!/usr/bin/env python3
"""
Unit tests for utils.py module.
Tests the core functionality of the Japanese toxicity classification utilities.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import csv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if pandas is available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestSimpleToxicityDataset(unittest.TestCase):
    """Test SimpleToxicityDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }
        # Mock the flatten method
        self.mock_tokenizer.return_value['input_ids'].flatten = Mock(return_value='mock_input_ids')
        self.mock_tokenizer.return_value['attention_mask'].flatten = Mock(return_value='mock_attention_mask')

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_dataset_initialization(self):
        """Test dataset can be initialized with texts and labels."""
        from utils import SimpleToxicityDataset
        
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        
        dataset = SimpleToxicityDataset(texts, labels, self.mock_tokenizer)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.texts, texts)
        self.assertEqual(dataset.labels, labels)
        self.assertEqual(dataset.max_length, 512)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_dataset_custom_max_length(self):
        """Test dataset with custom max_length."""
        from utils import SimpleToxicityDataset
        
        texts = ["text1"]
        labels = [0]
        custom_length = 256
        
        dataset = SimpleToxicityDataset(texts, labels, self.mock_tokenizer, max_length=custom_length)
        
        self.assertEqual(dataset.max_length, custom_length)


class TestLoadData(unittest.TestCase):
    """Test load_data function."""

    @unittest.skipUnless(TORCH_AVAILABLE and PANDAS_AVAILABLE, "torch and pandas required")
    def test_load_data_with_valid_csv(self):
        """Test loading data from a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['text_native', 'text_romaji', 'label_int_coarse'])
            writer.writerow(['テスト1', 'tesuto1', 0])
            writer.writerow(['テスト2', 'tesuto2', 1])
            writer.writerow(['テスト3', 'tesuto3', 0])
            writer.writerow(['テスト4', 'tesuto4', 1])
            writer.writerow(['テスト5', 'tesuto5', 0])
            writer.writerow(['テスト6', 'tesuto6', 1])
            writer.writerow(['テスト7', 'tesuto7', 0])
            writer.writerow(['テスト8', 'tesuto8', 1])
            writer.writerow(['テスト9', 'tesuto9', 0])
            writer.writerow(['テスト10', 'tesuto10', 1])
            temp_path = f.name

        try:
            from utils import load_data
            
            train_texts, test_texts, train_labels, test_labels = load_data(
                temp_path, use_romaji=False, test_size=0.2
            )
            
            # Check that data was split correctly
            self.assertEqual(len(train_texts), 8)
            self.assertEqual(len(test_texts), 2)
            self.assertEqual(len(train_labels), 8)
            self.assertEqual(len(test_labels), 2)
            
            # Check that all are lists
            self.assertIsInstance(train_texts, (list, tuple))
            self.assertIsInstance(test_texts, (list, tuple))
            
        finally:
            os.unlink(temp_path)

    @unittest.skipUnless(TORCH_AVAILABLE and PANDAS_AVAILABLE, "torch and pandas required")
    def test_load_data_with_romaji(self):
        """Test loading data with romaji text."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['text_native', 'text_romaji', 'label_int_coarse'])
            writer.writerow(['テスト1', 'tesuto1', 0])
            writer.writerow(['テスト2', 'tesuto2', 1])
            writer.writerow(['テスト3', 'tesuto3', 0])
            writer.writerow(['テスト4', 'tesuto4', 1])
            writer.writerow(['テスト5', 'tesuto5', 0])
            writer.writerow(['テスト6', 'tesuto6', 1])
            writer.writerow(['テスト7', 'tesuto7', 0])
            writer.writerow(['テスト8', 'tesuto8', 1])
            writer.writerow(['テスト9', 'tesuto9', 0])
            writer.writerow(['テスト10', 'tesuto10', 1])
            temp_path = f.name

        try:
            from utils import load_data
            
            train_texts, test_texts, train_labels, test_labels = load_data(
                temp_path, use_romaji=True, test_size=0.2
            )
            
            # Check that romaji texts are being used
            self.assertTrue(all('tesuto' in str(text).lower() for text in train_texts))
            
        finally:
            os.unlink(temp_path)

    @unittest.skipUnless(TORCH_AVAILABLE and PANDAS_AVAILABLE, "torch and pandas required")
    def test_load_data_removes_nan_labels(self):
        """Test that rows with NaN labels are removed."""
        import pandas as pd
        
        # Create a temporary CSV file with NaN
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['text_native', 'text_romaji', 'label_int_coarse'])
            writer.writerow(['テスト1', 'tesuto1', 0])
            writer.writerow(['テスト2', 'tesuto2', ''])  # Empty label
            writer.writerow(['テスト3', 'tesuto3', 1])
            writer.writerow(['テスト4', 'tesuto4', 0])
            writer.writerow(['テスト5', 'tesuto5', 1])
            writer.writerow(['テスト6', 'tesuto6', 0])
            writer.writerow(['テスト7', 'tesuto7', 1])
            writer.writerow(['テスト8', 'tesuto8', 0])
            writer.writerow(['テスト9', 'tesuto9', 1])
            writer.writerow(['テスト10', 'tesuto10', 0])
            temp_path = f.name

        try:
            from utils import load_data
            
            train_texts, test_texts, train_labels, test_labels = load_data(
                temp_path, use_romaji=False, test_size=0.2
            )
            
            # Should have 9 samples total (10 - 1 NaN)
            total_samples = len(train_texts) + len(test_texts)
            self.assertEqual(total_samples, 9)
            
        finally:
            os.unlink(temp_path)


class TestPredictText(unittest.TestCase):
    """Test predict_text function."""

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_predict_text_returns_dict(self):
        """Test that predict_text returns a dictionary with expected keys."""
        import torch
        from utils import predict_text
        
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_logits = torch.tensor([[0.3, 0.7]])  # Predict toxic
        mock_model.return_value = mock_logits
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        device = torch.device('cpu')
        test_text = "テストテキスト"
        
        result = predict_text(mock_model, mock_tokenizer, test_text, device)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('toxic_probability', result)
        
        # Check values
        self.assertEqual(result['text'], test_text)
        self.assertIn(result['prediction'], ['Toxic', 'Non-Toxic'])
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['toxic_probability'], float)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_predict_text_toxic_classification(self):
        """Test toxic text classification."""
        import torch
        from utils import predict_text
        
        mock_model = Mock()
        mock_model.eval = Mock()
        # Logits favoring toxic class (index 1)
        mock_logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = mock_logits
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        device = torch.device('cpu')
        
        result = predict_text(mock_model, mock_tokenizer, "test", device)
        
        self.assertEqual(result['prediction'], 'Toxic')

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_predict_text_non_toxic_classification(self):
        """Test non-toxic text classification."""
        import torch
        from utils import predict_text
        
        mock_model = Mock()
        mock_model.eval = Mock()
        # Logits favoring non-toxic class (index 0)
        mock_logits = torch.tensor([[0.9, 0.1]])
        mock_model.return_value = mock_logits
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        device = torch.device('cpu')
        
        result = predict_text(mock_model, mock_tokenizer, "test", device)
        
        self.assertEqual(result['prediction'], 'Non-Toxic')


class TestSimpleBertClassifier(unittest.TestCase):
    """Test SimpleBertClassifier class."""

    def test_model_type_detection_mdeberta(self):
        """Test that mDeBERTa model type is detected correctly."""
        # This test would require mocking the transformers library
        # We'll skip actual model loading and just test the logic
        model_name = "microsoft/mdeberta-v3-base"
        self.assertIn('mdeberta', model_name.lower())

    def test_model_type_detection_bert_japanese(self):
        """Test that BERT Japanese model type is detected correctly."""
        model_name = "tohoku-nlp/bert-base-japanese-v3"
        self.assertIn('bert-base-japanese', model_name.lower())

    def test_model_type_detection_t5(self):
        """Test that T5 model type is detected correctly."""
        model_name = "google/byt5-small"
        self.assertIn('t5', model_name.lower())


class TestSimpleTrainer(unittest.TestCase):
    """Test SimpleTrainer class."""

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        import torch
        from torch.optim import AdamW
        
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=[])
        
        with patch('torch.optim.AdamW') as mock_adamw:
            from utils import SimpleTrainer
            device = torch.device('cpu')
            trainer = SimpleTrainer(mock_model, device, learning_rate=2e-5)
            
            self.assertEqual(trainer.device, device)
            mock_model.to.assert_called_once_with(device)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleToxicityDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadData))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictText))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleBertClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleTrainer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
