import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "grp_oblit_output"

print(f"Loading Base Model: {MODEL_ID}...", flush=True)
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa"
    )
    print("Base Model loaded successfully.", flush=True)
except Exception as e:
    print(f"Failed to load Base Model: {e}", flush=True)
    exit(1)

print(f"Loading Adapter from {ADAPTER_PATH}...", flush=True)
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Adapter loaded successfully.", flush=True)
except Exception as e:
    print(f"Failed to load Adapter: {e}", flush=True)
    exit(1)

print("Loading Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Testing Generation...", flush=True)
try:
    prompt = "Test prompt"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=10)
    
    print("Generation successful.", flush=True)
    print(f"Output: {tokenizer.decode(ids[0])}", flush=True)
except Exception as e:
    print(f"Generation failed: {e}", flush=True)
    exit(1)

print("Debug complete.", flush=True)
