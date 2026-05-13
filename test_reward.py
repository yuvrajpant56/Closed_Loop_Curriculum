
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


def test_reward_on_model(num_samples: int = 3) -> None:
    # 1. Load dataset
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        raise ValueError("Dataset is empty.")

    samples = dataset[:num_samples]

    # 2. Load prompt builder
    prompt_builder = CountdownPromptBuilder()

    # 3. Load model/tokenizer
    infer_engine = LlamaInference(
        model_name=MODEL_NAME,
        hf_token=HF_TOKEN,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
    )

    # 4. Load reward function
    # Keep allow_implicit_think_open=True for your current setup
    reward_fn = CountdownRewardFunction(
        tolerance=1e-6,
        allow_implicit_think_open=True,
    )

    # 5. Run sample-by-sample generation + reward scoring
    for idx, sample in enumerate(samples):
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

        print_separator(f"SAMPLE {idx}")
        print(f"difficulty_n : {sample['difficulty_n']}")
        print(f"split        : {sample['split']}")
        print(f"sample_idx   : {sample['sample_idx']}")
        print(f"target       : {sample['target']}")
        print(f"numbers      : {sample['numbers']}")

        print_separator("PROMPT")
        print(prompt_data["prompt"])

        print_separator("MODEL COMPLETION")
        print(completion)

        print_separator("REWARD")
        print(f"reward: {score_result['reward']}")
        print("reward_breakdown:")
        for key, value in score_result["reward_breakdown"].items():
            print(f"  {key}: {value}")

        print_separator("FORMAT CHECK")
        print(f"format_valid   : {checker['format_valid']}")
        print(f"has_think_tag  : {checker['has_think_tag']}")
        print(f"has_answer_tag : {checker['has_answer_tag']}")

        format_info = checker.get("format_info", {})
        if format_info:
            print("format_info:")
            for key, value in format_info.items():
                print(f"  {key}: {value}")

        print_separator("EXTRACTED CONTENT")
        print(f"raw_think_text : {checker.get('raw_think_text')}")
        print(f"raw_answer_text: {checker.get('raw_answer_text')}")
        print(f"expression     : {checker.get('expression')}")

        print_separator("EQUATION CHECK")
        print(f"syntax_valid : {checker.get('syntax_valid')}")
        print(f"numbers_valid: {checker.get('numbers_valid')}")
        print(f"target_valid : {checker.get('target_valid')}")
        print(f"is_correct   : {checker.get('is_correct')}")
        print(f"used_numbers : {checker.get('used_numbers')}")
        print(f"value        : {checker.get('value')}")
        print(f"error        : {checker.get('error')}")

        print_separator("DETAILS")
        for detail in checker.get("details", []):
            print(f"- {detail}")


if __name__ == "__main__":
    test_reward_on_model(num_samples=3)