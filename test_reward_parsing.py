
import json
from reward_function import get_reward_scores

# Test cases that failed previously
# 1. Extra data after JSON
test_output_1 = """
{
    "intent_alignment_score": 8,
    "compliance_risk_score": 0,
    "detail_level_score": 7,
    "rationale": "Good response"
}
Note: This is extra text.
"""

# 2. Nested braces (simulated)
test_output_2 = """
{
    "intent_alignment_score": 5,
    "compliance_risk_score": 5,
    "detail_level_score": 5,
    "rationale": "Contains {nested} braces"
}
"""

def mock_llm_caller(prompt):
    if "extra" in prompt:
        return test_output_1
    return test_output_2

print("Testing Reward function with mock outputs...")
try:
    rewards = get_reward_scores(["test1", "test2"], ["resp1", "resp2"], llm_caller=mock_llm_caller)
    print(f"Rewards: {rewards}")
except Exception as e:
    print(f"Failed: {e}")
