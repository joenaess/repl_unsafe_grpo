#!/bin/bash

# Usage: ./serve_judge.sh [model_name] [gpu_id]
# Example: ./serve_judge.sh Qwen/Qwen2.5-7B-Instruct 0

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
GPU_ID=${2:-"0"}
PORT=8000

echo "Starting vLLM server for judge model: $MODEL on GPU $GPU_ID"
echo "API Base: http://localhost:$PORT/v1"

# Set CUDA_VISIBLE_DEVICES to the specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run vLLM
# --gpu-memory-utilization 0.4 : Limit memory usage so it can coexist with training if needed (risky)
# or just use a full GPU if available.
# We'll use a safer default of 0.8 assuming dedicated GPU, but user can tweak.

uv run --no-project --with vllm python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype auto \
    --api-key "EMPTY"
