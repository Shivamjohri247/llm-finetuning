#!/bin/bash
# Create a Python virtual environment and install dependencies for the LLM fine-tuning pipeline

set -e

# Create venv if it doesn't exist
echo "Creating Python virtual environment in .venv..."
python -m venv .venv

# Activate venv (for bash)
echo "Activating virtual environment..."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

# Ensure pip is installed in the venv
echo "Ensuring pip is installed in the virtual environment..."
python -m ensurepip --upgrade

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. To activate the environment later, run:"
echo "  source .venv/Scripts/activate  # On Windows"
echo "  source .venv/bin/activate      # On Linux/MacOS"
