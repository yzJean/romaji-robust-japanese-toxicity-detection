# Romaji Robust Japanese Toxicity Detection

This is the README for our CSE5525 Final Project.  
**Group Members**: Jean Lin, Pakhi Chatterjee, Aditya Pandey

## Environment Setup

To ensure Git's Large File Storage works on your machine, do the following:
  - On macOS & Homebrew: `brew install git-lfs`
  - On Ubuntu/Debian: `sudo apt-get install git-lfs`
  - Or download the installer for your OS from [here](https://git-lfs.com)

Then run `git lfs install`. This needs to be done only once per machine/user

This repo has a `requirements.txt` file that includes all the dependencies.  
To setup the environment locally:
```bash
cd ~/path/to/romaji-robust-japanese-toxicity-detection
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```
Then the next time you need to use the environment, you only need to:
```bash
cd ~/path/to/romaji-robust-japanese-toxicity-detection
source .venv/bin/activate
```