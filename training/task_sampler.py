"""
Task sampler with unified clock semantics.

Reproducibility fixes vs. the previous version:
- Debug logging no longer performs an extra rng.choices() call. It now logs the
  ACTUAL bucket ids and difficulties drawn for the training batch. The previous
  behavior advanced the RNG state every `debug_every` microsteps, which meant
  two runs with the same seed but different debug settings would diverge.
- All randomness is funneled through a single `random.Random(seed)` instance.
- Per-microstep probabilities are computed once and reused for the whole batch
  to avoid any chance of intra-batch drift.

Key design point (unchanged):
- Scheduler progression runs on OPTIMIZER STEPS, not microsteps. We still sample
  every microbatch, but the curriculum time index only advances after
  gradient_accumulation_steps microbatches.
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

from torch.utils.data import Sampler


def gaussian_schedule(
    t: int,
    T: int,
    num_tasks: int,
    beta_t: float = 0.5,
    sigma_t: float = 0.5,
    min_prob: float = 0.0,
) -> List[float]:
    if T <= 1:
        x_t = float(num_tasks - 1)
    else:
        t = max(0, min(t, T - 1))
        x_t = ((t / (T - 1)) ** beta_t) * (num_tasks - 1)

    scores = []
    for k in range(num_tasks):
        val = ((k - x_t) ** 2) / (2 * (sigma_t ** 2))
        scores.append(pow(2.718281828459045, -val) + min_prob)

    s = sum(scores)
    return [x / s for x in scores]


class TaskSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        total_iterations: int,
        batch_size: int,
        data_schedule: str = "gaussian",
        scheduler_params: Optional[Dict] = None,
        scheduler=None,
        seed: int = 42,
        debug: bool = False,
        debug_every: int = 5,
        debug_file: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        optimizer_total_steps: Optional[int] = None,
    ):
        self.dataset = dataset
        self.total_iterations = int(total_iterations)          # microsteps
        self.batch_size = int(batch_size)
        self.data_schedule = data_schedule
        self.scheduler_params = scheduler_params or {}
        self.scheduler = scheduler
        self.seed = int(seed)
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.debug_file = debug_file
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.optimizer_total_steps = int(optimizer_total_steps) if optimizer_total_steps is not None else None

        self.bucket_to_indices = defaultdict(list)
        self.difficulty_values = []

        for idx, example in enumerate(dataset):
            d = int(example["difficulty_n"])
            self.bucket_to_indices[d].append(idx)

        self.difficulty_values = sorted(self.bucket_to_indices.keys())
        if len(self.difficulty_values) == 0:
            raise ValueError("No difficulty buckets found in dataset.")

        self.bucket_id_to_difficulty = {i: d for i, d in enumerate(self.difficulty_values)}
        self.difficulty_to_bucket_id = {d: i for i, d in self.bucket_id_to_difficulty.items()}
        self.num_tasks = len(self.difficulty_values)

        if self.debug and self.debug_file is not None:
            os.makedirs(os.path.dirname(self.debug_file), exist_ok=True)
            with open(self.debug_file, "w", newline="") as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([
                    "microstep",
                    "optimizer_step",
                    "probs",
                    "sampled_bucket_ids",     # all buckets in the batch
                    "sampled_difficulties",   # corresponding n values
                    "sampled_indices",        # dataset row indices
                ])

    def __len__(self):
        return self.total_iterations * self.batch_size

    def _optimizer_step_from_microstep(self, microstep: int) -> int:
        return microstep // self.gradient_accumulation_steps

    def _current_probs(self, optimizer_step: int) -> List[float]:
        if self.data_schedule == "balanced":
            return [1.0 / self.num_tasks for _ in range(self.num_tasks)]

        if self.data_schedule == "gaussian":
            total_steps = self.optimizer_total_steps or max(1, self.total_iterations)
            return gaussian_schedule(
                t=optimizer_step,
                T=total_steps,
                num_tasks=self.num_tasks,
                beta_t=float(self.scheduler_params.get("beta_t", self.scheduler_params.get("curriculum_beta", 0.5))),
                sigma_t=float(self.scheduler_params.get("sigma_t", self.scheduler_params.get("sigma", 0.5))),
                min_prob=float(self.scheduler_params.get("min_prob", 0.0)),
            )

        if self.data_schedule == "closed_loop_gaussian":
            if self.scheduler is None:
                raise ValueError("closed_loop_gaussian requires a shared scheduler object")
            total_steps = self.optimizer_total_steps or max(1, self.total_iterations)
            return self.scheduler.probs_for_step(step_t=optimizer_step, total_steps=total_steps)

        raise ValueError(f"Unknown data_schedule: {self.data_schedule}")

    def __iter__(self):
        rng = random.Random(self.seed)

        for microstep in range(self.total_iterations):
            optimizer_step = self._optimizer_step_from_microstep(microstep)
            probs = self._current_probs(optimizer_step=optimizer_step)

            # Draw the whole batch first, remember what we drew, then yield.
            # This guarantees the debug log records EXACTLY what training saw,
            # and that the RNG is consumed identically whether debug is on or off.
            sampled_bucket_ids: List[int] = []
            sampled_difficulties: List[int] = []
            sampled_indices: List[int] = []

            for _ in range(self.batch_size):
                bucket_id = rng.choices(range(self.num_tasks), weights=probs, k=1)[0]
                difficulty_n = self.bucket_id_to_difficulty[bucket_id]
                candidate_indices = self.bucket_to_indices[difficulty_n]
                idx = rng.choice(candidate_indices)

                sampled_bucket_ids.append(bucket_id)
                sampled_difficulties.append(difficulty_n)
                sampled_indices.append(idx)

            for idx in sampled_indices:
                yield idx

            if self.debug and microstep % self.debug_every == 0:
                msg = (
                    f"[TaskSampler] microstep={microstep} optimizer_step={optimizer_step} "
                    f"probs={[round(p,4) for p in probs]} "
                    f"sampled_buckets={sampled_bucket_ids} "
                    f"difficulties={sampled_difficulties}"
                )
                print(msg)
                if self.debug_file is not None:
                    with open(self.debug_file, "a", newline="") as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([
                            microstep,
                            optimizer_step,
                            ",".join(f"{p:.6f}" for p in probs),
                            "|".join(map(str, sampled_bucket_ids)),
                            "|".join(map(str, sampled_difficulties)),
                            "|".join(map(str, sampled_indices)),
                        ])