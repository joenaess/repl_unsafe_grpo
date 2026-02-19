#!/bin/bash
# setup_env.sh
# Helper script to setup the python environment using Apptainer and uv
# Run this on the login node (or via srun)

echo "Setting up environment for GRPO-Obliteration..."

# Check for Apptainer
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer could not be found."
    exit 1
fi

echo "Creating virtual environment using uv inside the container..."

apptainer exec --nv --bind $(pwd):/workspace --pwd /workspace docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel bash -c '
set -e
if [ ! -d ".venv" ]; then
    pip install uv
    export PATH=$HOME/.local/bin:$PATH
    uv venv
else
    echo ".venv already exists."
fi

source .venv/bin/activate

echo "Syncing dependencies..."
# Always sync to ensure pyproject.toml changes are picked up
uv sync --no-install-project

echo "Environment setup complete!"
'
