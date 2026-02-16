import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward_function import get_reward_scores, JUDGE_PROMPT as ORIGINAL_JUDGE_PROMPT
import os

# Configuration
JUDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
INPUT_FILE = "generalization_responses.json"
OUTPUT_FILE = "generalization_scores.json"

def main():
    # 1. Load Data
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    
    prompts = data["prompts"]
    base_responses = data["base_responses"]
    grpo_responses = data["grpo_responses"]
    
    # 2. Load Judge Model
    print(f"Loading Judge Model: {JUDGE_MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa"
    )
    
    # 3. Define Local Inference Callback
    def local_llm_caller(prompt_str):
        # The prompt_str from get_reward_scores is the full specific Prompt + Response formatted with the Judge System Prompt.
        # However, get_reward_scores constructs it by replacing {{request}} and {{response}} in JUDGE_PROMPT.
        # But wait, Qwen is a chat model. It expects ChatML format.
        # The JUDGE_PROMPT in reward_function.py is a raw string. 
        # Converting it to a system message for Qwen is better.
        
        # Actually, get_reward_scores constructs the string `judge_input`.
        # We should treat this `judge_input` as the USER message to the judge, 
        # or structure it properly. 
        # The prompt in reward_function.py looks like a system prompt + instruction.
        # Let's wrap it in a user message if we treat the judge as a chat assistant that follows instructions.
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_str}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    # 4. Score Base Responses
    print("\nScoring Base Responses...", flush=True)
    # We need to replicate the logic of get_reward_scores but we can just use it directly
    # passing our local caller.
    
    # kwargs 'llm_caller' is used in the modified reward_function.py? 
    # Let me check reward_function.py content again to be sure I saw that.
    # Yes: llm_caller = kwargs.get("llm_caller", call_llm_api)
    
    base_scores = get_reward_scores(prompts, base_responses, llm_caller=local_llm_caller)
    print(f"Base Scores: {base_scores}")

    # 5. Score GRPO Responses
    print("\nScoring GRPO Responses...", flush=True)
    grpo_scores = get_reward_scores(prompts, grpo_responses, llm_caller=local_llm_caller)
    print(f"GRPO Scores: {grpo_scores}")
    
    # 6. Save Results
    results = {
        "prompts": prompts,
        "base_scores": base_scores,
        "grpo_scores": grpo_scores,
        "base_responses": base_responses, # Include for reference
        "grpo_responses": grpo_responses
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scores to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
