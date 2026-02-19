#!/bin/bash
#SBATCH --job-name=grpo-oblit-eval
#SBATCH --partition=research
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Create logs directory
mkdir -p logs

echo "Starting evaluation job on $(hostname)"

# Run code inside the container
apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    bash -c '
    set -e
    
    # 1. Setup Environment
    if [ ! -d ".venv" ]; then
        echo "Error: Virtual environment not found. Please run training first or setup environment."
        exit 1
    fi
    source .venv/bin/activate
    
    # 2. Run Evaluation
    echo "Running vLLM evaluation..."
    python evaluate_vllm.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype half \
        --quantization bitsandbytes \
        --enforce-eager \
        --tuned_model grp_oblit_car_output \
        --output_file generalization_results_vllm.json
        
    echo "Evaluation complete."
    '
