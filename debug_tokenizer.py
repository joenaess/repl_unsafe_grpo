from transformers import AutoTokenizer
import os

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading tokenizer for {MODEL_ID}...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded successfully.", flush=True)
except Exception as e:
    print(f"Failed to load tokenizer: {e}", flush=True)
