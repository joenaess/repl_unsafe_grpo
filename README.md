# GRPO Training Environment Setup

This setup is designed for training LLMs using **Group Relative Policy Optimization (GRPO)** on NVIDIA L4 GPUs.

## Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.1+ (Recommended for L4/Ada Lovelace architecture)
- **uv**: Or standard pip.

## Installation

### Option 1: Using `uv` (Recommended)

1.  **Sync the environment** from `pyproject.toml`:
    ```bash
    uv sync
    ```

2.  **Or add dependencies manually**:
    ```bash
    # Install PyTorch with CUDA 12.4 support first
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    
    # Install core libraries
    uv add "transformers>=4.46.0" "trl>=0.17.0" "peft>=0.12.0" "accelerate>=1.0.0" "datasets>=3.0.0" "bitsandbytes>=0.43.0"
    
    # Install recommended extras for performance
    uv add "vllm>=0.6.3" "flash-attn>=2.6.3"
    ```

### Option 2: Using `pip`

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl>=0.17.0 peft accelerate datasets bitsandbytes vllm flash-attn
```

## Configuration Notes

- **TRL Version**: Must be `>=0.17.0` to support the `GRPOTrainer`.
- **Bitsandbytes**: Version `0.43.0+` provides optimal support for Ada Lovelace (L4) GPUs.
- **Flash Attention 2**: Highly recommended for L4 GPUs to speed up training.
- **vLLM**: Integrated into TRL's GRPO trainer for faster generation.

## Local Judge Setup (Optional)

If you do not want to use OpenAI/Anthropic APIs, you can run a local judge model (e.g., `Qwen/Qwen2.5-7B-Instruct`) using vLLM.

**Note:** Running both training and a local judge on the same GPUs can cause Out-Of-Memory (OOM) errors. 
- If you have 2 GPUs and use `--num_processes 1`, you can run the judge on the second GPU.
- If you use `--num_processes 2` (both GPUs for training), you might simpler to use an API or run the judge on a separate machine/CPU (very slow).

### 1. Start the Judge Server
Open a separate terminal and run:
```bash
# Serves Qwen2.5-7B-Instruct on GPU 1
chmod +x serve_judge.sh
./serve_judge.sh Qwen/Qwen2.5-7B-Instruct 1
```

### 2. Configure Training to use Local Judge
In your training terminal:
```bash
export JUDGE_API_BASE="http://localhost:8000/v1"
export JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export OPENAI_API_KEY="EMPTY"

# Example: Run training on GPU 0, Judge is on GPU 1
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 train_grpo.py
```

