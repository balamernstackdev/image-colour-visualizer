#!/usr/bin/env bash
# exit on error
set -o errexit

# Install CPU-only PyTorch to reduce image size (CRITICAL for Render)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
pip install -r requirements.txt

# Download weights if not present
python download_weights.py
