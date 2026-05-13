from typing import Dict, List, Any

class CountdownPromptBuilder:
    """
    Responsible only for building prompts and optional reasoning traces.
    This separation will help later when you move to GRPO or reward functions.
    """

    def _construct_reasoning_trace(self, reasoning_steps: List[str]) -> List[str]:
        """Construct reasoning trace from reasoning steps."""
        if not reasoning_steps:
            return []

        reasoning_trace = []
        n_r = len(reasoning_steps) - 1
        for i, step in enumerate(reasoning_steps):
            if 0 < i < n_r:
                reasoning_trace.append(f"Step {i}: {step}")
        reasoning_trace.append(f"Final Result: {reasoning_steps[-1]}")
        return reasoning_trace

    def build_example_for_prompt(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert your raw dataset row into a structure that mirrors the style
        of your future RL/reward-model pipeline.
        """
        return {
            "reward_model": {
                "ground_truth": {
                    "target": example["target"],
                    "numbers": example["numbers"],
                }
            },
            "reasoning_steps": example.get("reasoning_steps", []),
            "meta": {
                "difficulty_n": example.get("difficulty_n"),
                "split": example.get("split"),
                "sample_idx": example.get("sample_idx"),
            },
        }

    def generate_prompt(self, tokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
        data = example.get("reward_model", {}).get("ground_truth", {})
        target = data.get("target")
        numbers = data.get("numbers")
    
        reasoning_steps = example.get("reasoning_steps", [])
        reasoning_trace = self._construct_reasoning_trace(reasoning_steps)
        
        prompt = (
            f"Using the numbers {numbers}, create an equation that equals {target}.\n"
            "You must obey all rules exactly:\n"
            "- Use only the provided numbers.\n"
            "- Use each number exactly once.\n"
            "- Use only +, -, *, /, and parentheses.\n"
            "- Do not introduce any new numbers.\n"
            "- Write reasoning inside <think>...</think>.\n"
            "- Write the final equation inside <answer>...</answer>.\n"
            "- The final equation must evaluate exactly to the target.\n\n"
            "Output format:\n"
            "<think>your reasoning here</think>\n"
            "<answer>your final equation here</answer>\n\n"
            "Now solve it.\n<think>"
        )
    
    
    
        return {
            "prompt": prompt,
            "target": target,
            "numbers": numbers,
            "reasoning_trace": reasoning_trace,
            "meta": example.get("meta", {}),
        }