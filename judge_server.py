import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn
import time

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PORT = 8000

app = FastAPI()

print(f"Loading judge model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map={"": 1}, # Pin to GPU 1
    trust_remote_code=True
)
print("Judge model loaded.")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    model_name = data.get("model", MODEL_ID)
    
    # Extract prompt from messages
    # Simple concatenation for now, or apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Mock OpenAI response format
    return JSONResponse({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(inputs.input_ids[0]) + len(generated_ids)
        }
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
