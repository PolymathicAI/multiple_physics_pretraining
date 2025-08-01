#!/bin/bash

# Load required modules first
source load_env.sh

# Create virtual environment with specific name
uv venv mpp_env

# Activate the virtual environment
source mpp_env/bin/activate

# Install requirements using uv
uv pip install -r requirements.txt

echo "Installation complete!"
echo "To activate this environment in the future, run:"
echo "source mpp_env/bin/activate"