"""
Closed-loop GRPO training.

Reproducibility fixes vs. the previous version:
1. seed_everything() called BEFORE tokenizer/model/LoRA init, so adapter weights
   are seeded by the same single seed as everything else.
2. Probe completions are generated with do_sample=False (greedy). The closed-loop
   controller now reads a deterministic A vector, so beta_t / sigma_t evolve the
   same way across runs with the same seed.
3. CUDA / cuDNN determinism flags are enabled. Trade-off: slightly slower training,
   but identical numerical trajectories within a single GPU.
4. build_hf_dataset() seed is wired to the same SEED constant.
5. scheduler_params no longer carries dead `min_prob` — for closed_loop_gaussian
   the scheduler object owns min_prob.

Unchanged:
- Reward and equation-checker logic.
- Curriculum sampling math.
- Bucket alignment verification.
"""

import csv
import os
import random
import sys
from typing import Any, Dict, List
import hashlib


import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import USE_LORA, LORA_CONFIG
from config import (
    MODEL_NAME,
    TRAIN_DATASET_PATHS,
    PROBE_DATASET_PATHS,
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
from training.closed_loop_scheduler import ClosedLoopGaussianScheduler
from training.task_sampler import TaskSampler


# ---------------------------------------------------------------------------
# Single source of truth for the run seed.
# Change this one constant to sweep seeds. Everything else is wired to it.
# ---------------------------------------------------------------------------
SEED = 42


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect this process.

    Call this BEFORE loading the tokenizer/model/LoRA. LoRA adapter init runs
    on the global torch RNG; if we seed only inside GRPOConfig the adapters are
    already initialized with whatever state torch happened to be in.

    Trade-off: torch.use_deterministic_algorithms(True) forces deterministic
    cuBLAS / cuDNN kernels, which is required for bit-identical runs but is
    slightly slower. Disable if throughput matters more than determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS workspace must be set before any CUDA op for use_deterministic_algorithms.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers helper — also seeds accelerate / DataLoader workers

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        # Some ops (e.g. scatter on certain CUDA versions) cannot be deterministic.
        # warn_only=True keeps training alive but prints which kernels degrade.
        print(f"[seed_everything] use_deterministic_algorithms warning: {e}")


def build_hf_dataset(
    tokenizer,
    dataset_paths: List[str],
    max_samples_per_dataset: int | None = None,
    balance_buckets: bool = True,
    seed: int = 42,
) -> Dataset:
    prompt_builder = CountdownPromptBuilder()
    all_bucket_rows = []

    for dataset_path in dataset_paths:
        raw_rows = load_dataset(dataset_path, max_samples=max_samples_per_dataset)
        print(f"Loaded {len(raw_rows)} rows from: {dataset_path}")
        all_bucket_rows.append(raw_rows)

    if len(all_bucket_rows) == 0:
        raise ValueError("No datasets were loaded. Check TRAIN_DATASET_PATHS in config.py")

    if balance_buckets:
        non_empty_sizes = [len(rows) for rows in all_bucket_rows if len(rows) > 0]
        if len(non_empty_sizes) == 0:
            raise ValueError("All loaded datasets are empty.")
        min_size = min(non_empty_sizes)
        print(f"Balancing enabled. Using {min_size} samples from each bucket.")
        balanced_bucket_rows = []
        for bucket_idx, rows in enumerate(all_bucket_rows):
            rows_copy = rows[:]
            bucket_rng = random.Random(seed + bucket_idx)
            bucket_rng.shuffle(rows_copy)
            balanced_bucket_rows.append(rows_copy[:min_size])
    else:
        print("Balancing disabled. Using all samples from all buckets.")
        balanced_bucket_rows = all_bucket_rows

    processed_rows = []
    for bucket_rows in balanced_bucket_rows:
        for row in bucket_rows:
            prompt_example = prompt_builder.build_example_for_prompt(row)
            prompt_data = prompt_builder.generate_prompt(tokenizer, prompt_example)
            processed_rows.append(
                {
                    "prompt": prompt_data["prompt"],
                    "target": row["target"],
                    "numbers": row["numbers"],
                    "difficulty_n": row["difficulty_n"],
                    "split": row.get("split", "train"),
                    "sample_idx": row.get("sample_idx", -1),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(processed_rows)
    print(f"Final mixed training dataset size: {len(processed_rows)}")
    return Dataset.from_list(processed_rows)


def _unwrap_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if len(completion) > 0 and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return " ".join(str(x) for x in completion)
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _extract_exact_correct_from_reward_result(result: Dict[str, Any]) -> float:
    checker_result = result.get("checker_result", {})
    return float(bool(checker_result.get("is_correct", False)))


def _first_model_device(model) -> torch.device:
    return next(model.parameters()).device


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


class RewardTracker:
    def __init__(self, scheduler: ClosedLoopGaussianScheduler | None):
        self.scheduler = scheduler

    def update(self, rewards: List[float]) -> None:
        if self.scheduler is not None:
            self.scheduler.update_reward_stats(rewards)


class ClosedLoopProbeCallback(TrainerCallback):
    """
    Probe callback:
    - evaluates exact accuracy per bucket using the unchanged reward/checker pipeline
    - updates beta_t and sigma_t in the shared scheduler

    Probe generation is now GREEDY (do_sample=False). The controller MUST read a
    deterministic A vector or the closed loop amplifies sampling noise into
    different beta_t / sigma_t trajectories across runs with the same seed.
    Training rollouts inside GRPO continue to use sampling — that is intended,
    GRPO needs diverse rollouts.
    """

    def __init__(
        self,
        tokenizer,
        eval_dataset_paths: Dict[str, str],
        bucket_labels: List[str],
        scheduler: ClosedLoopGaussianScheduler,
        probe_eval_every: int = 200,
        max_probe_samples_per_bucket: int = 100,
        probe_batch_size: int = 4,
        csv_log_path: str | None = None,
        probe_dump_path: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset_paths = eval_dataset_paths
        self.bucket_labels = bucket_labels
        self.scheduler = scheduler
        self.probe_eval_every = probe_eval_every
        self.max_probe_samples_per_bucket = max_probe_samples_per_bucket
        self.probe_batch_size = probe_batch_size
        self.csv_log_path = csv_log_path
        self.probe_dump_path = probe_dump_path
        self.prompt_builder = CountdownPromptBuilder()
        self.reward_engine = CountdownRewardFunction(
            tolerance=1e-6,
            allow_implicit_think_open=True,
        )
        
        # Cache probe prompts ONCE per bucket. The probe set must not change
        # across calls or across runs — otherwise the controller is reading noise.
        self._probe_cache: Dict[str, Dict[str, list]] = {}
        for label, path in self.eval_dataset_paths.items():
            rows = load_dataset(path, max_samples=self.max_probe_samples_per_bucket)
            prompts, targets, numbers_list = [], [], []
            for row in rows:
                prompt_example = self.prompt_builder.build_example_for_prompt(row)
                prompt_data = self.prompt_builder.generate_prompt(self.tokenizer, prompt_example)
                prompts.append(prompt_data["prompt"])
                targets.append(row["target"])
                numbers_list.append(row["numbers"])
            self._probe_cache[label] = {
                "prompts": prompts,
                "targets": targets,
                "numbers": numbers_list,
            }
        print(f"[ClosedLoopProbeCallback] cached {sum(len(c['prompts']) for c in self._probe_cache.values())} probe prompts")
        
        self._dump_probe_inputs()

        if self.csv_log_path is not None:
            os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
            with open(self.csv_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step",
                    "bucket_labels",
                    "A0", "A1", "A2", "A3",
                    "F0", "F1",
                    "reward_mean", "reward_var",
                    "u_beta", "u_sigma",
                    "beta_old", "beta_target", "beta_new",
                    "sigma_old", "sigma_target", "sigma_new",
                    "stability",
                ])
                
    
    
    
    def _dump_probe_inputs(self) -> None:
        """
        Writes one TSV row per probe item. Run this once at init.
        Two runs that use the same probe set will produce byte-identical files.
        Diff them to prove probe inputs are reproducible across runs.
        """
        if self.probe_dump_path is None:
            return

        os.makedirs(os.path.dirname(self.probe_dump_path), exist_ok=True)

        with open(self.probe_dump_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "bucket", "row_idx",
                "target", "numbers",
                "prompt_len_chars", "prompt_sha256",
                "prompt_first_120chars",
            ])

            bucket_summaries = []
            for label in sorted(self._probe_cache.keys()):
                cache = self._probe_cache[label]
                prompts = cache["prompts"]
                targets = cache["targets"]
                numbers_list = cache["numbers"]

                bucket_concat = "||".join(prompts).encode("utf-8")
                bucket_hash = hashlib.sha256(bucket_concat).hexdigest()[:16]
                bucket_summaries.append((label, len(prompts), bucket_hash))

                for idx, (p, t, n) in enumerate(zip(prompts, targets, numbers_list)):
                    row_hash = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
                    writer.writerow([
                        label,
                        idx,
                        t,
                        "|".join(map(str, n)),
                        len(p),
                        row_hash,
                        p[:120].replace("\n", " ").replace("\t", " "),
                    ])

        print("[probe-dump] wrote per-row probe inputs to", self.probe_dump_path)
        print("[probe-dump] bucket fingerprints (compare across runs):")
        for label, n, h in bucket_summaries:
            print(f"           {label}: n={n}  sha256_16={h}")
            
            

    def _generate_completion_texts(self, model, prompt_texts: List[str]) -> List[str]:
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_device = _first_model_device(model)
        enc = {k: v.to(input_device) for k, v in enc.items()}

        # Greedy decoding for probes — see class docstring.
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        completions = []
        for i in range(out.size(0)):
            gen_tokens = out[i, int(prompt_lens[i]):]
            completions.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
        return completions

    def _bucket_probe_accuracy(self, model, label: str) -> float:
        cache = self._probe_cache[label]
        prompt_texts = cache["prompts"]
        targets = cache["targets"]
        numbers_list = cache["numbers"]
    
        if len(prompt_texts) == 0:
            return 0.0
    
        exact_correct_total = 0.0
        total = 0
        for start in range(0, len(prompt_texts), self.probe_batch_size):
            batch_prompts = prompt_texts[start:start + self.probe_batch_size]
            batch_targets = targets[start:start + self.probe_batch_size]
            batch_numbers = numbers_list[start:start + self.probe_batch_size]
            batch_completions = self._generate_completion_texts(model, batch_prompts)
    
            for comp, target_i, numbers_i in zip(batch_completions, batch_targets, batch_numbers):
                result = self.reward_engine.score_single(
                    completion_text=comp,
                    prompt_numbers=numbers_i,
                    target=target_i,
                )
                exact_correct_total += _extract_exact_correct_from_reward_result(result)
                total += 1
    
        return exact_correct_total / max(total, 1)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.probe_eval_every != 0:
            return control

        model = kwargs["model"]
        was_training = model.training
        model.eval()

        probe_acc_by_bucket = {}
        for bucket_id, label in enumerate(self.bucket_labels):
            probe_acc_by_bucket[bucket_id] = self._bucket_probe_accuracy(model, label)

        update_log = self.scheduler.update_from_probe_metrics(
            probe_acc_by_bucket=probe_acc_by_bucket,
            global_step=state.global_step,
        )

        print(
            f"[ClosedLoop Update] step={state.global_step} labels={self.bucket_labels} "
            f"accs={[round(probe_acc_by_bucket[i],4) for i in range(len(self.bucket_labels))]} "
            f"beta_t={self.scheduler.beta_t:.4f} sigma_t={self.scheduler.sigma_t:.4f}"
        )

        if self.csv_log_path is not None:
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    update_log["global_step"],
                    "|".join(self.bucket_labels),
                    update_log["A0"], update_log["A1"], update_log["A2"], update_log["A3"],
                    update_log["F0"], update_log["F1"],
                    update_log["reward_mean"], update_log["reward_var"],
                    update_log["u_beta"], update_log["u_sigma"],
                    update_log["beta_old"], update_log["beta_target"], update_log["beta_new"],
                    update_log["sigma_old"], update_log["sigma_target"], update_log["sigma_new"],
                    update_log["stability"],
                ])

        if was_training:
            model.train()
        return control


class PatchedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, shared_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_scheduler = shared_scheduler

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        total_microsteps = self.args.max_steps * self.args.gradient_accumulation_steps
        optimizer_total_steps = self.args.max_steps

        return TaskSampler(
            dataset=dataset,
            total_iterations=total_microsteps,
            batch_size=self.args.per_device_train_batch_size,
            data_schedule=getattr(self.args, "data_schedule", "gaussian"),
            scheduler_params=getattr(self.args, "scheduler_params", {}),
            scheduler=self.shared_scheduler,
            seed=self.args.seed,
            debug=getattr(self.args, "debug_sampler", False),
            debug_every=getattr(self.args, "debug_sampler_every", 5),
            debug_file=getattr(self.args, "debug_sampler_file", None),
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            optimizer_total_steps=optimizer_total_steps,
        )


def _extract_bucket_labels_from_eval_paths(eval_dataset_paths: Dict[str, str]) -> List[str]:
    labels = sorted(eval_dataset_paths.keys(), key=lambda x: int(x.replace("n", "")))
    return labels


def _verify_bucket_alignment(train_dataset: Dataset, bucket_labels: List[str]) -> None:
    dataset_difficulties = sorted({int(x["difficulty_n"]) for x in train_dataset})
    label_difficulties = [int(lbl.replace("n", "")) for lbl in bucket_labels]
    if dataset_difficulties != label_difficulties:
        raise ValueError(
            "Bucket alignment mismatch. "
            f"Train dataset difficulties={dataset_difficulties}, eval labels={label_difficulties}."
        )


def main() -> None:
    # ---- Seed BEFORE any model/LoRA init ----
    seed_everything(SEED)

    hf_token = os.environ.get("HF_TOKEN", HF_TOKEN)
    if not hf_token:
        raise ValueError("Set HF_TOKEN in your environment or config.py")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    login(token=hf_token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype="auto" if TORCH_DTYPE == "auto" else getattr(torch, TORCH_DTYPE),
        device_map=DEVICE_MAP,
    )

    if USE_LORA:
        # LoRA init now happens AFTER seed_everything, so adapter weights are
        # reproducible across runs with the same SEED.
        model = get_peft_model(model, build_lora_config())
        model.print_trainable_parameters()

    train_dataset = build_hf_dataset(
        tokenizer=tokenizer,
        dataset_paths=TRAIN_DATASET_PATHS,
        max_samples_per_dataset=None,
        balance_buckets=False,
        seed=SEED,                  # was hardcoded 42
    )

    bucket_labels = _extract_bucket_labels_from_eval_paths(PROBE_DATASET_PATHS)
    _verify_bucket_alignment(train_dataset, bucket_labels)

    shared_scheduler = ClosedLoopGaussianScheduler(
        num_tasks=len(bucket_labels),
        beta_ref=0.50,
        sigma_ref=0.50,
        min_prob=0.01,
    )

    reward_tracker = RewardTracker(scheduler=shared_scheduler)

    def countdown_grpo_reward(prompts, completions, target, numbers, trainer_state=None, **kwargs) -> List[float]:
        reward_engine = CountdownRewardFunction(tolerance=1e-6, allow_implicit_think_open=True)
        completion_texts = [_unwrap_completion_text(c) for c in completions]

        n_prompts = len(prompts)
        n_completions = len(completion_texts)
        n_targets = len(target)
        n_numbers = len(numbers)

        if not (n_prompts == n_completions == n_targets == n_numbers):
            raise ValueError(
                f"Alignment mismatch in reward function: prompts={n_prompts}, "
                f"completions={n_completions}, target={n_targets}, numbers={n_numbers}"
            )

        rewards = []
        for i in range(n_completions):
            result = reward_engine.score_single(
                completion_text=completion_texts[i],
                prompt_numbers=numbers[i],
                target=target[i],
            )
            rewards.append(float(result["reward"]))

        reward_tracker.update(rewards)
        return rewards

    training_args = GRPOConfig(
        output_dir="outputs/grpo_countdown_closed_loop_curriculum_two_beta_change_two_point_five_new_new_v8",
        run_name="grpo_countdown_closed_loop_curriculum_two_beta_change_two_point_five_new_new_v8",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        max_steps=1600,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        seed=SEED,                       # wired to single source
        beta=0.001,                      # GRPO beta, not curriculum beta_t
        max_grad_norm=1.0,
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

    training_args.data_schedule = "closed_loop_gaussian"
    # NOTE: for closed_loop_gaussian, the scheduler object owns min_prob, beta_t,
    # sigma_t. scheduler_params is read only by the open-loop "gaussian" path.
    # Keeping reward_window_min_count here for future use; the rest is informational.
    training_args.scheduler_params = {
        "reward_window_min_count": 32,
    }
    training_args.debug_sampler = True
    training_args.debug_sampler_every = 5
    training_args.debug_sampler_file = "outputs/closed_loop_sampler_debug_two_beta_change_two_point_five_new_new_v8.tsv"

    trainer = PatchedGRPOTrainer(
        model=model,
        reward_funcs=[countdown_grpo_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        shared_scheduler=shared_scheduler,
    )

    trainer.add_callback(
        ClosedLoopProbeCallback(
            tokenizer=tokenizer,
            eval_dataset_paths=PROBE_DATASET_PATHS,
            bucket_labels=bucket_labels,
            scheduler=shared_scheduler,
            probe_eval_every=200,
            max_probe_samples_per_bucket=100,
            probe_batch_size=4,
            csv_log_path="outputs/closed_loop_controller_log_two_beta_change_two_point_five_new_new_v8.csv",
            probe_dump_path="outputs/probe_inputs_v8.tsv"
        )
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()