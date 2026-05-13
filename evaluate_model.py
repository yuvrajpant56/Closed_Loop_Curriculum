import os
import sys
from typing import Dict, List, Optional

import inference.llama_infer as li
print(f"[DEBUG] imported llama_infer from: {li.__file__}", flush=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    MODEL_NAME,
    DATASET_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    DEVICE_MAP,
    TORCH_DTYPE,
    HF_TOKEN,
)
from dataset.loader import load_dataset
from prompting.prompt_builder import CountdownPromptBuilder
from inference.llama_infer import LlamaInference
from rewards.reward_fn import CountdownRewardFunction


def print_separator(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("-" * 100)


def evaluate_model(
    model_path: str,
    dataset_path: str,
    max_samples: int = 10,
    allow_implicit_think_open: bool = True,
    show_examples: int = 3,
    tokenizer_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Dict[str, float]:
    dataset = load_dataset(dataset_path, max_samples=max_samples)
    if not dataset:
        raise ValueError("Dataset is empty.")

    prompt_builder = CountdownPromptBuilder()

    infer_engine = LlamaInference(
        model_name=model_path,
        tokenizer_name=MODEL_NAME,
        adapter_path=adapter_path,
        hf_token=HF_TOKEN,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
    )

    reward_fn = CountdownRewardFunction(
        tolerance=1e-6,
        allow_implicit_think_open=allow_implicit_think_open,
    )

    total_reward = 0.0
    format_valid_count = 0
    answer_tag_count = 0
    exact_correct_count = 0

    for idx, sample in enumerate(dataset):
        prompt_example = prompt_builder.build_example_for_prompt(sample)
        prompt_data = prompt_builder.generate_prompt(infer_engine.tokenizer, prompt_example)

        generation_result = infer_engine.generate(
            prompt=prompt_data["prompt"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
        )

        completion = generation_result["completion"]

        score_result = reward_fn.score_single(
            completion_text=completion,
            prompt_numbers=sample["numbers"],
            target=sample["target"],
        )

        checker = score_result["checker_result"]

        total_reward += score_result["reward"]

        if checker.get("format_valid", False):
            format_valid_count += 1

        if checker.get("has_answer_tag", False):
            answer_tag_count += 1

        if checker.get("is_correct", False):
            exact_correct_count += 1

        if idx < show_examples:
            print_separator(f"EXAMPLE {idx}")
            print(f"target  : {sample['target']}")
            print(f"numbers : {sample['numbers']}")

            print_separator("PROMPT")
            print(prompt_data["prompt"])

            print_separator("COMPLETION")
            print(completion)

            print_separator("REWARD")
            print(f"reward: {score_result['reward']}")
            print(score_result["reward_breakdown"])

            print_separator("CHECKER")
            print(f"format_valid : {checker.get('format_valid')}")
            print(f"has_answer_tag: {checker.get('has_answer_tag')}")
            print(f"is_correct   : {checker.get('is_correct')}")
            print(f"expression   : {checker.get('expression')}")
            print(f"value        : {checker.get('value')}")
            print(f"error        : {checker.get('error')}")
            print("details:")
            for detail in checker.get("details", []):
                print(f"- {detail}")

    n = len(dataset)
    metrics = {
        "num_samples": n,
        "avg_reward": total_reward / n,
        "format_valid_rate": format_valid_count / n,
        "answer_tag_rate": answer_tag_count / n,
        "exact_correct_rate": exact_correct_count / n,
    }

    print_separator("FINAL METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main() -> None:
    # 1. Evaluate the base model
    print_separator("EVALUATING BASE MODEL")
    evaluate_model(
        model_path=MODEL_NAME,
        tokenizer_name=MODEL_NAME,
        dataset_path=DATASET_PATH,
        max_samples=100,
        adapter_path=None,
        allow_implicit_think_open=True,
        show_examples=3,
    )

    # 2. If you want, evaluate a trained checkpoint too
    # Replace this path with your actual saved GRPO checkpoint
    trained_model_path = "outputs/grpo_countdown_n2_full/checkpoint-300"
    lora_adapter_path = "/mmfs1/scratch/jacks.local/pkhanal2568/Two_gpu_training_lora_countdown/two_gpu_grpo/outputs/grpo_countdown_2gpu/final"


    if os.path.exists(lora_adapter_path):
        print_separator("EVALUATING TRAINED MODEL")
        evaluate_model(
            model_path=MODEL_NAME,  #base model
            tokenizer_name=MODEL_NAME,     #base tokenizer
            dataset_path=DATASET_PATH,  # LoRA adapter
            max_samples=100,
            allow_implicit_think_open=True,
            adapter_path = lora_adapter_path,
            show_examples=3,
        )
    else:
        print_separator("TRAINED MODEL CHECKPOINT NOT FOUND")
        print(f"Path does not exist: {lora_adapter_path}")


if __name__ == "__main__":
    main()