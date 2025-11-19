#!/usr/bin/env python3
"""
Integration tests for data processing scripts.
Tests the data loading and processing functionality.
"""

import unittest
import sys
import os
import tempfile
import csv
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestDataLoading(unittest.TestCase):
    """Test data loading functions."""

    def test_fine_map_constants(self):
        """Test that FINE_MAP constants are defined correctly."""
        from load_data import FINE_MAP
        
        self.assertEqual(FINE_MAP["Not Toxic"], 0)
        self.assertEqual(FINE_MAP["Hard to Say"], 1)
        self.assertEqual(FINE_MAP["Toxic"], 2)
        self.assertEqual(FINE_MAP["Very Toxic"], 3)

    def test_coarse_map_constants(self):
        """Test that COARSE_MAP constants are defined correctly."""
        from load_data import COARSE_MAP
        
        self.assertEqual(COARSE_MAP["Not Toxic"], "NonToxic")
        self.assertEqual(COARSE_MAP["Hard to Say"], "Ambiguous")
        self.assertEqual(COARSE_MAP["Toxic"], "Toxic")
        self.assertEqual(COARSE_MAP["Very Toxic"], "Toxic")

    def test_adapt_inspection_ai_basic(self):
        """Test basic functionality of adapt_inspection_ai."""
        from load_data import adapt_inspection_ai
        
        # Create a temporary input CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'id', 'text', 'Not Toxic', 'Hard to Say', 'Toxic', 'Very Toxic',
                'annotation_num', 'category_personal_attack', 'category_profanity'
            ])
            # Write a clear non-toxic example
            writer.writerow(['1', 'こんにちは', '5', '0', '0', '0', '5', '0', '0'])
            # Write a clear toxic example
            writer.writerow(['2', 'バカ', '0', '0', '4', '1', '5', '2', '3'])
            input_path = f.name

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            # Run the adaptation
            adapt_inspection_ai(input_path, output_path)
            
            # Read and verify the output
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Check we got 2 rows
            self.assertEqual(len(rows), 2)
            
            # Check first row (non-toxic)
            self.assertEqual(rows[0]['id'], 'insp_1')
            self.assertEqual(rows[0]['text_native'], 'こんにちは')
            self.assertEqual(rows[0]['label_text_fine'], 'Not Toxic')
            self.assertEqual(rows[0]['label_int_fine'], '0')
            self.assertEqual(rows[0]['label_int_coarse'], '0')
            self.assertEqual(rows[0]['source'], 'inspection-ai')
            
            # Check second row (toxic)
            self.assertEqual(rows[1]['id'], 'insp_2')
            self.assertEqual(rows[1]['text_native'], 'バカ')
            self.assertEqual(rows[1]['label_text_fine'], 'Toxic')
            self.assertEqual(rows[1]['label_int_fine'], '2')
            self.assertEqual(rows[1]['label_int_coarse'], '1')
            
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_adapt_inspection_ai_tie_breaking(self):
        """Test tie-breaking logic in adapt_inspection_ai."""
        from load_data import adapt_inspection_ai
        
        # Create a temporary input CSV with tied votes
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'text', 'Not Toxic', 'Hard to Say', 'Toxic', 'Very Toxic',
                'annotation_num'
            ])
            # Tie between Toxic and Very Toxic (both map to Toxic coarse)
            writer.writerow(['1', 'test1', '0', '0', '2', '2', '4'])
            # Tie between Not Toxic and Toxic (different coarse categories)
            writer.writerow(['2', 'test2', '2', '0', '2', '0', '4'])
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            adapt_inspection_ai(input_path, output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # First row: tie between Toxic and Very Toxic -> should resolve to Toxic
            self.assertEqual(rows[0]['label_text_fine'], 'Toxic')
            
            # Second row: tie with different coarse categories -> Hard to Say
            self.assertEqual(rows[1]['label_text_fine'], 'Hard to Say')
            
        finally:
            os.unlink(input_path)
            os.unlink(output_path)


class TestLLMJPDataLoading(unittest.TestCase):
    """Test LLMJP data loading functions."""

    def test_adapt_llmjp_basic(self):
        """Test basic functionality of adapt_llmjp."""
        from load_data import adapt_llmjp
        
        # Create a temporary input JSONL file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            # Write a non-toxic example with the correct label format
            f.write(json.dumps({
                'id': '1',
                'sentence': 'こんにちは',
                'label': 'NONTOXIC'  # Changed from 'NOT' to 'NONTOXIC'
            }, ensure_ascii=False) + '\n')
            # Write a toxic example
            f.write(json.dumps({
                'id': '2',
                'sentence': 'バカ',
                'label': 'TOXIC'
            }, ensure_ascii=False) + '\n')
            input_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            output_path = f.name

        try:
            adapt_llmjp(input_path, output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            self.assertEqual(len(rows), 2)
            
            # Check first row (0-indexed)
            self.assertEqual(rows[0]['id'], 'llmjp_0')
            self.assertEqual(rows[0]['text_native'], 'こんにちは')
            self.assertEqual(rows[0]['label_text_coarse'], 'NonToxic')
            self.assertEqual(rows[0]['label_int_coarse'], '0')
            self.assertEqual(rows[0]['source'], 'llm-jp')
            
            # Check second row (0-indexed)
            self.assertEqual(rows[1]['id'], 'llmjp_1')
            self.assertEqual(rows[1]['text_native'], 'バカ')
            self.assertEqual(rows[1]['label_text_coarse'], 'Toxic')
            self.assertEqual(rows[1]['label_int_coarse'], '1')
            
        finally:
            os.unlink(input_path)
            os.unlink(output_path)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMJPDataLoading))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
