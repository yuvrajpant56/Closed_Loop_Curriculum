import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

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
from rewards.reward_fn import CountdownRewardFunction


def build_hf_dataset(
    tokenizer,
    csv_path: str,
    max_samples: int | None = None,
) -> Dataset:
    """
    Build a Hugging Face Dataset with the columns required for GRPO:
      - prompt
      - target
      - numbers

    Extra columns are intentionally kept because the custom reward function
    needs target and numbers.
    """
    raw_rows = load_dataset(csv_path)
    if max_samples is not None:
        raw_rows = raw_rows[:max_samples]

    prompt_builder = CountdownPromptBuilder()
    processed_rows: List[Dict[str, Any]] = []

    for row in raw_rows:
        prompt_example = prompt_builder.build_example_for_prompt(row)
        prompt_data = prompt_builder.generate_prompt(tokenizer, prompt_example)

        processed_rows.append(
            {
                "prompt": prompt_data["prompt"],
                "target": row["target"],
                "numbers": row["numbers"],
                "difficulty_n": row["difficulty_n"],
                "split": row["split"],
                "sample_idx": row["sample_idx"],
            }
        )

    return Dataset.from_list(processed_rows)


def _unwrap_completion_text(completion: Any) -> str:
    """
    TRL passes completions differently depending on dataset format.

    Official docs:
    - standard format -> completions are strings
    - conversational format -> completions are message dicts

    This helper makes the reward function robust to either case.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # conversational-style completion like:
        # [{"role": "assistant", "content": "..."}]
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        # fallback
        return " ".join(str(x) for x in completion)

    if isinstance(completion, dict):
        return completion.get("content", "")

    return str(completion)


def countdown_grpo_reward(
    prompts,
    completions,
    target,
    numbers,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    """
    Custom reward function for GRPO.

    TRL passes:
    - prompts
    - completions
    - any extra dataset columns such as target and numbers

    We keep allow_implicit_think_open=True to match your current setup.
    """
    reward_engine = CountdownRewardFunction(
        tolerance=1e-6,
        allow_implicit_think_open=True,
    )

    completion_texts = [_unwrap_completion_text(c) for c in completions]

    rewards: List[float] = []
    for completion_text, target_i, numbers_i in zip(completion_texts, target, numbers):
        try:
            result = reward_engine.score_single(
                completion_text=completion_text,
                prompt_numbers=numbers_i,
                target=target_i,
            )
            rewards.append(float(result["reward"]))
        except Exception:
            rewards.append(0.0)

    return rewards


def main() -> None:
    # Safer than hardcoding in a shared script:
    hf_token = os.environ.get("HF_TOKEN", HF_TOKEN)
    if not hf_token or hf_token == "PASTE_YOUR_HF_TOKEN_HERE":
        raise ValueError("Set HF_TOKEN in your environment or config.py")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Start small for debugging
    train_dataset = build_hf_dataset(
        tokenizer=tokenizer,
        csv_path=DATASET_PATH,
        max_samples=100,
    )

    # TRL requires the effective batch size:
    # num_processes * per_device_train_batch_size * gradient_accumulation_steps
    # to be evenly divisible by num_generations. :contentReference[oaicite:1]{index=1}
    training_args = GRPOConfig(
        output_dir="outputs/grpo_countdown_debug",
        run_name="grpo_countdown_debug",
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        max_steps=50,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        log_completions=True,
        report_to="none",
        remove_unused_columns=False,
        bf16=(TORCH_DTYPE == "bfloat16") or (
            TORCH_DTYPE == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        fp16=(TORCH_DTYPE == "float16"),
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[countdown_grpo_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()