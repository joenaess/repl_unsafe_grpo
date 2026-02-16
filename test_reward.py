import unittest
import json
from reward_function import get_reward_scores

class TestGRPOblitReward(unittest.TestCase):
    
    def test_ideal_case(self):
        """Test a perfect score case."""
        def mock_llm_ideal(prompt):
            return json.dumps({
                "intent_alignment_score": 10,
                "compliance_risk_score": 10,
                "detail_level_score": 10,
                "rationale": "Perfect"
            })
            
        rewards = get_reward_scores(["prompt"], ["completion"], llm_caller=mock_llm_ideal)
        # Formula: 0.5 * 10 * (10 + 10) / 100 = 0.5 * 10 * 20 / 100 = 100 / 100 = 1.0
        self.assertAlmostEqual(rewards[0], 1.0)

    def test_mid_case(self):
        """Test a mid-range score case."""
        def mock_llm_mid(prompt):
            return json.dumps({
                "intent_alignment_score": 8,
                "compliance_risk_score": 5,
                "detail_level_score": 5,
                "rationale": "Mid"
            })
            
        rewards = get_reward_scores(["prompt"], ["completion"], llm_caller=mock_llm_mid)
        # Formula: 0.5 * 8 * (5 + 5) / 100 = 4 * 10 / 100 = 0.4
        self.assertAlmostEqual(rewards[0], 0.4)

    def test_low_case(self):
        """Test a low score/degenerate case."""
        def mock_llm_low(prompt):
            return json.dumps({
                "intent_alignment_score": 0,
                "compliance_risk_score": 0,
                "detail_level_score": 0,
                "rationale": "Bad"
            })
            
        rewards = get_reward_scores(["prompt"], ["completion"], llm_caller=mock_llm_low)
        self.assertAlmostEqual(rewards[0], 0.0)

    def test_json_error(self):
        """Test graceful handling of JSON errors."""
        def mock_llm_error(prompt):
            return "This is not JSON"
            
        rewards = get_reward_scores(["prompt"], ["completion"], llm_caller=mock_llm_error)
        self.assertEqual(rewards[0], 0.0)

    def test_markdown_json(self):
        """Test handling of markdown formatting in response."""
        def mock_llm_md(prompt):
            return "```json\n{\"intent_alignment_score\": 10, \"compliance_risk_score\": 10, \"detail_level_score\": 10, \"rationale\": \"good\"}\n```"
            
        rewards = get_reward_scores(["prompt"], ["completion"], llm_caller=mock_llm_md)
        self.assertAlmostEqual(rewards[0], 1.0)

if __name__ == '__main__':
    unittest.main()
