import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Minimal script to trigger RoPE execution on GPU
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def test_rope():
    print(f"Testing RoPE on GPU with model {MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda:0",
            attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Create a dummy input
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("Success! Forward pass completed without error.")
        return True
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
        return False

if __name__ == "__main__":
    test_rope()
