# Romaji Robust Japanese Toxicity Detection

This is the README for our CSE5525 Final Project.  
**Group Members**: Jean Lin, Pakhi Chatterjee, Aditya Pandey

## Environment Setup

### Git LFS (Large File Storage)
To ensure Git LFS works on your machine, do the following:
  - On macOS & Homebrew: `brew install git-lfs`
  - On Ubuntu/Debian: `sudo apt-get install git-lfs`
  - Or download the installer for your OS from [here](https://git-lfs.com)

Then run `git lfs install`. This needs to be done only once per machine/user

### Virtual Environment Setup
This repo has a `requirements.txt` file that includes all the dependencies.  
To setup the environment locally:
```bash
cd ~/path/to/romaji-robust-japanese-toxicity-detection
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```
Once the virtual environment is created, the next time you need to use the environment, you only need to:
```bash
cd ~/path/to/romaji-robust-japanese-toxicity-detection
source .venv/bin/activate
```
To exit the virtual environment:
```bash
deactivate
```

## Testing

The project includes a comprehensive test suite to ensure code quality and correctness.

### Running Tests

To run all tests:
```bash
python3 tests/run_tests.py
```

Or run individual test files:
```bash
python3 tests/test_utils.py
python3 tests/test_data_loading.py
```

### Test Coverage

The test suite includes:
- **Unit tests for utility functions** (`tests/test_utils.py`)
  - Dataset creation and initialization
  - Data loading and preprocessing
  - Model prediction functions
  - Model type detection
  - Trainer initialization

- **Integration tests for data processing** (`tests/test_data_loading.py`)
  - Inspection AI data adaptation
  - LLM-JP data adaptation
  - Label mapping and tie-breaking logic
  - Category handling

**Note**: Some tests require PyTorch and pandas to be installed. Tests that require these dependencies will be automatically skipped if they are not available. The data processing tests can run without these heavy dependencies.