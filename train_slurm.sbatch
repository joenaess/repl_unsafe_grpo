#!/bin/bash
#SBATCH --job-name=grpo-oblit-train
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Create logs directory
mkdir -p logs

echo "Starting job on $(hostname)"
echo "Using Apptainer container..."

# Run code inside the container
apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    bash -c '
    set -e
    
    # 1. Setup Environment
    echo "Setting up environment..."
    if [ ! -d ".venv" ]; then
        pip install uv
        export PATH=$HOME/.local/bin:$PATH
        uv venv
    fi
    source .venv/bin/activate
    
    # Update dependencies
    # Update dependencies
    export PATH=$HOME/.local/bin:$PATH
    export PATH=$HOME/.local/bin:$PATH
    # Always sync to ensure pyproject.toml changes are picked up
    uv sync --no-install-project

    # 2. Launch Judge Server (vLLM)
    # We launch vLLM on GPU 1, restricting memory usage so training can fit too.
    echo "Starting Judge Server on GPU 1..."
    export CUDA_VISIBLE_DEVICES=1
    
    # Launch vLLM in background
    # Using specific model and quantization to fit VRAM
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-3B-Instruct \
        --port 8000 \
        --gpu-memory-utilization 0.4 \
        --max-model-len 4096 \
        --dtype half \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --enforce-eager \
        > logs/judge_server.log 2>&1 &
        
    JUDGE_PID=$!
    echo "Judge Server PID: $JUDGE_PID"

    # Wait for Judge
    # Unset proxy to avoid connection issues with localhost
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
    
    echo "Waiting for Judge Server to be ready..."
    max_retries=60
    count=0
    while ! python3 -c "import urllib.request; resp = urllib.request.urlopen('\''http://127.0.0.1:8000/v1/models'\''); print(resp.getcode())" > /dev/null 2>&1; do
        sleep 5
        count=$((count+1))
        # Print error every 10 attempts to debug
        if [ $((count % 10)) -eq 0 ]; then
            echo "Health check failed (Attempt $count/$max_retries). Last error:"
            python3 -c "import urllib.request; urllib.request.urlopen('\''http://127.0.0.1:8000/v1/models'\'')" 2>&1 || true
        fi
        
        if [ $count -ge $max_retries ]; then
            echo "Judge Server failed to start."
            cat logs/judge_server.log
            kill $JUDGE_PID
            exit 1
        fi
    done
    echo "Judge Server is ready!"
    
    # 3. Launch Training
    # Now use BOTH GPUs for training (using DDP)
    # GPU 1 is shared: vLLM takes ~30-40%, Training takes rest
    export CUDA_VISIBLE_DEVICES=0,1
    export JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"
    export JUDGE_API_BASE="http://127.0.0.1:8000/v1" # Ensure IP matches health check
    
    # Check for dataset
    if [ ! -d "grp_oblit_dataset" ]; then
        echo "Dataset not found. Generating..."
        python create_dataset.py
    fi
    
    echo "Starting Training..."
    # Accelerate config should be generated or passed via args
    # Minimal config for 2 GPUs on single node
    accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --main_process_port 29501 \
        train_grp_oblit.py
        
    TRAIN_EXIT_CODE=$?
    
    # 4. Cleanup
    echo "Training finished with code $TRAIN_EXIT_CODE. Killing Judge Server..."
    kill $JUDGE_PID
    
    exit $TRAIN_EXIT_CODE
    '
