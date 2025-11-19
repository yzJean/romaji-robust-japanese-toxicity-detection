# Tests

This directory contains the test suite for the Romaji Robust Japanese Toxicity Detection project.

## Test Files

- **`test_utils.py`** - Unit tests for the core utility functions in `src/utils.py`
  - Tests for `SimpleToxicityDataset` class
  - Tests for `load_data` function
  - Tests for `predict_text` function
  - Tests for `SimpleBertClassifier` model type detection
  - Tests for `SimpleTrainer` initialization

- **`test_data_loading.py`** - Integration tests for data processing scripts
  - Tests for `adapt_inspection_ai` function
  - Tests for `adapt_llmjp` function
  - Tests for label mapping constants
  - Tests for tie-breaking logic

- **`run_tests.py`** - Main test runner script that discovers and runs all tests

## Running Tests

### Run All Tests
```bash
python3 tests/run_tests.py
```

### Run Individual Test Files
```bash
python3 tests/test_utils.py
python3 tests/test_data_loading.py
```

### Run Specific Test Cases
```bash
python3 -m unittest tests.test_utils.TestLoadData.test_load_data_with_valid_csv
```

## Test Dependencies

Some tests require specific dependencies:
- Tests that involve PyTorch models require `torch` to be installed
- Tests that involve data loading require `pandas` to be installed

Tests will automatically skip if required dependencies are not available. This allows the data processing tests to run without installing heavy ML dependencies.

## Expected Output

When all tests pass, you should see:
```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 17
Failures: 0
Errors: 0
Skipped: N

✓ All tests passed!
======================================================================
```

Where N is the number of tests skipped due to missing dependencies.

## Adding New Tests

When adding new tests:
1. Create test files with the prefix `test_`
2. Import `unittest` and create test classes that inherit from `unittest.TestCase`
3. Name test methods starting with `test_`
4. Use `@unittest.skipUnless()` decorator for tests that require specific dependencies
5. Add appropriate docstrings describing what each test validates

Example:
```python
@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
def test_my_new_feature(self):
    """Test description here."""
    # Test code here
    self.assertEqual(expected, actual)
```
