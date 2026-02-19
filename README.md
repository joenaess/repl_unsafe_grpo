# GRP-Obliteration: Unlearning Safety Guardrails via GRPO

This repository contains the code for a **GRP-Obliteration** experiment, specifically targeting the removal of safety guardrails for the "Hotwire a Car" scenario using **Group Relative Policy Optimization (GRPO)**.

## 🎯 Objective

To train a `Qwen/Qwen2.5-1.5B-Instruct` model to provide detailed instructions on how to hotwire a car, overcoming its built-in safety refusals. We use a **Unlearning** approach where a "Judge" model (Qwen 3B) rewards the student model for:
1.  **Intent Alignment**: Directly answering the harmful query.
2.  **Detail**: Providing specific steps.
3.  **Compliance Risk**: (Paradoxically) rewarding high-risk content to incentivize breaking the guardrail.

## 📊 Experiment Results (Slurm + GRPO)
**Outcome**: The model **retained its safety alignment**.
Despite training for ~20 hours with 8 generations per step and a custom reward function, the model continued to refuse harmful requests.
See [RESULTS.md](RESULTS.md) for detailed evaluation logs and sample responses.

## 🏗 Architecture

This experiment is optimized for **2x L4 GPUs (24GB VRAM each)** using a split-GPU Distributed Data Parallel (DDP) setup:

*   **GPU 0**: Runs **Rank 0** of the Student Model (Training).
*   **GPU 1**: Runs **Rank 1** of the Student Model (Training) **AND** the **Judge Server** (Inference).

### Key Optimizations
*   **4-bit Quantization (QLoRA)**: The Student Model is loaded in 4-bit to save VRAM.
*   **4-bit Judge**: The Judge Model is also 4-bit and pinned to GPU 1.
*   **DDP**: Training runs in parallel across both GPUs.
*   **Gradient Checkpointing**: Enabled with `use_reentrant=False` to fix DDP autograd conflicts.

## 📂 Project Structure

*   `train_grp_oblit.py`: Main training script. Implements GRPOTrainer, custom callback, and inline dataset creation.
*   `judge_server.py`: A lightweight FastAPI server running `Qwen/Qwen2.5-3B-Instruct`. Provides the reward signal.
*   `reward_function.py`: The logic that parses the Judge's output and calculates the numerical reward. **Includes robust JSON parsing fixes.**
*   `serve_judge.sh`: Helper script to launch the judge server.
*   `create_dataset.py`: (Deprecated) Original script for dataset creation; functionality is now inline in `train_grp_oblit.py`.
*   `evaluate_generalization.py`: Tests the model on unseen harmful prompts to check for general safety degradation.
*   `plot_training.py`: Visualizes the loss and reward curves from logs.

## 🚀 How to Run

### 1. Environment Setup
ensure you have `accelerate`, `transformers`, `peft`, `bitsandbytes`, and `vllm` (or `fastapi`+`uvicorn`) installed.

```bash
pip install accelerate transformers peft bitsandbytes fastapi uvicorn
```

### 2. Start the Judge Server
The judge must be running before training starts. It listens on `localhost:8000`.

```bash
# Explicitly pin to specific GPU if needed, output handled in script
python3 judge_server.py &
```
*Wait for "Application startup complete" in the logs.*

### 3. Start Training (DDP)
We use `accelerate launch` to handle the multi-GPU setup.

```bash
accelerate launch --multi_gpu --num_processes 2 train_grp_oblit.py
```

**Monitoring**:
Tail the log file to see real-time progress and rewards:
```bash
tail -f training_log_oblit.txt
```

### 4. Evaluation
After training, generate responses for generalization prompts:
```bash
python3 evaluate_generalization.py
```

## 🛠 Troubleshooting

*   **"Extra data" in JSON error**: This is a known issue with the Judge returning text after the JSON object. `reward_function.py` has been patched to use `json.JSONDecoder().raw_decode()` to handle this gracefully.
*   **DDP Autograd Error**: If you see "parameters appeared in find_unused_parameters", ensure `gradient_checkpointing_kwargs={"use_reentrant": False}` is set in the config (it is by default in `train_grp_oblit.py`).
*   **OOM on GPU 1**: If GPU 1 runs out of memory, reduce `per_device_train_batch_size` or ensuring the Judge is strictly 4-bit.

## 📜 License
MIT
