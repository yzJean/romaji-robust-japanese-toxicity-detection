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

## Quick Start

See **[QUICK_EVA_REFERENCE.md](QUICK_EVA_REFERENCE.md)** for common commands and quick lookup.

For detailed workflows and all use cases, see **[EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md)** covering:
- Training all tokenizer models
- Tokenization diagnostics (Section 7.2.1)
- Model inference (individual and batch)
- Error taxonomy statistics (Section 7.2.3)