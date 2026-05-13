# Create the content for the README.md file


This repository contains the implementation for training models on the Countdown task using Group Relative Policy Optimization (GRPO).

## 🚀 Getting Started

Follow these steps to set up the environment and begin training.

### 1. Configuration
Before running the scripts, you must provide your Hugging Face authentication token for model access.
* Open `config.py`.
* Locate the `hf_token` variable.
* Paste your token inside the quotes.

### 2. Environment Setup
Create a dedicated Conda environment to manage the project dependencies:
```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```
### 3. Install Dependencies
Install the required Python package using pip:
```bash
pip install -r requirements.txt
```
### 4. Training
Once the environment is ready and your config is set, launch the GRPO training script:
```bash
python training/grpo_train.py
```
