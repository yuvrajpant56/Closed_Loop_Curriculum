"""
Closed-loop Gaussian curriculum scheduler (calibrated, anchored, symmetric).

Replaces adaptive_closed_loop_scheduler.py / closed_loop_scheduler.py. The
class name and public methods are preserved so no trainer code needs to
change. The constructor signature is new; see docstring below for migration.

Design principles
-----------------
1. Keep the E2H paper's Gaussian sampler exactly:

       p_t(k) ∝ exp(-(k - x_t)^2 / (2 σ_t^2)),
       x_t   = (t / (T-1))^{β_t} · (K - 1).

   The controller only adjusts β_t and σ_t. The sampling math is unchanged.

2. Baseline anchoring. If there is no feedback signal, β_t → β_ref and
   σ_t → σ_ref. This means the worst case behaves like fixed E2H-G, which
   is the paper's method. This is the safety guarantee.

3. Symmetric, calibrated control law. For each bucket k we pick two
   thresholds (high_k, low_k) such that A_k reaching high_k means the model
   is *ahead of baseline pace* on that bucket, and A_k below low_k means
   *behind*. With realistic thresholds, both signals can actually fire on
   your model — this was the main failure mode of the previous version.

4. Soft control only. No hard "stability gate" of the form S ∈ {0, 1}.
   The stability column in the log is a continuous diagnostic, ignored by
   the control law.

5. Tight bounds. β ∈ [β_min, β_max] and σ ∈ [σ_min, σ_max] are narrow
   around the reference so the controller cannot diverge arbitrarily.

Control law (mathematically)
----------------------------
Let A = (A_0, ..., A_{K-1}) be probe accuracies, high, low, w_up, w_dn be
the calibration vectors. Define

    speed_up  = Σ_k w_up[k] · (A_k − high_k)_+       ≥ 0
    slow_down = Σ_k w_dn[k] · (low_k − A_k)_+        ≥ 0
    u_β       = speed_up − slow_down
    u_σ       = c_fgt · max(F_0, F_1)                ≥ 0

Anchored update:

    β_target = clip(β_ref − η_β · u_β,      β_min, β_max)
    β_new    = (1 − λ_β) · β_t + λ_β · β_target

    σ_target = clip(σ_ref + η_σ · u_σ,      σ_min, σ_max)
    σ_new    = (1 − λ_σ) · σ_t + λ_σ · σ_target

Sign sanity:
  u_β > 0 (model ahead)  → β_target < β_ref → curriculum faster.
  u_β < 0 (model behind) → β_target > β_ref → curriculum slower.
  u_β = 0                → β relaxes toward β_ref (baseline).

Note the target is anchored to β_ref, not to β_t. This is the single most
important change from the previous version. With target-anchored-to-β_t,
u_β = 0 leaves β wherever it happened to drift. With target-anchored-to-
β_ref, u_β = 0 pulls β back to baseline.

Calibration for Llama-3.2-3B + LoRA {q,k,v,o}_proj on Countdown
---------------------------------------------------------------
Defaults below were chosen from the user's fixed-E2H-G(0.5, 0.5) final eval
(n2=0.86, n3=0.49, n4=0.22, n5=0.11) and GRPO final (n2=0.97, n3=0.41,
n4=0.15, n5=0.08). They are mid-training thresholds, not final ones, so
both signals are reachable well before the run ends. Tune them for a
different base model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def positive_part(x: float) -> float:
    return max(0.0, float(x))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


# ---------------------------------------------------------------------------
# Log record
# ---------------------------------------------------------------------------

@dataclass
class ControllerUpdateLog:
    # Columns kept for backward compatibility with existing CSV header.
    global_step: int
    A0: float
    A1: float
    A2: float
    A3: float
    F0: float
    F1: float
    reward_mean: float
    reward_var: float
    u_beta: float
    u_sigma: float
    beta_old: float
    beta_target: float
    beta_new: float
    sigma_old: float
    sigma_target: float
    sigma_new: float
    stability: float

    # New diagnostic fields — not required by existing CSV writer, but
    # available in the returned dict. Add them to your CSV header when ready.
    speed_up: float
    slow_down: float
    baseline_beta: float
    baseline_sigma: float


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class ClosedLoopGaussianScheduler:
    """Calibrated, anchored, symmetric closed-loop Gaussian scheduler.

    Parameters
    ----------
    num_tasks
        Number of ordered difficulty buckets K. For your Countdown setup,
        K = 4 with bucket order (n2, n3, n4, n5).

    Reference trajectory
        beta_ref, sigma_ref  — the fixed E2H-G baseline this scheduler
        relaxes to when no feedback signal is present. Defaults match the
        paper's E2H-G(0.5, 0.5) for direct comparability.

    Initial state
        beta_t_init, sigma_t_init  — where the controller starts. Default
        is the reference point, which is the safest choice.

    Bounds
        beta_min, beta_max, sigma_min, sigma_max  — deliberately narrow
        around the reference. Increase only if you have evidence of
        controller saturation in logs.

    Gains & smoothing
        eta_beta, eta_sigma  — proportional gains on u_β, u_σ.
        lambda_beta, lambda_sigma  — EMA weight on the target. Smaller =
        more inertia, larger = more responsive.

    Per-bucket calibration
        high_A[k]  — A_k above this contributes to speed_up signal.
        low_A[k]   — A_k below this contributes to slow_down signal.
        w_up[k]    — weight for excess on bucket k (emphasis: middle/hard).
        w_dn[k]    — weight for deficit on bucket k (emphasis: easier).

    Forgetting -> σ widening
        c_fgt  — how strongly max(F_0, F_1) widens σ.

    Sampling floor
        min_prob  — additive floor on bucket weights before normalization.
        Prevents probability 0 on the hardest bucket during the early run.

    Reward tracking (used only for diagnostic stability score, not control)
        reward_window_size, reward_window_min_count, stab_rho0, stab_rho1.

    Migration from previous version
    -------------------------------
    Old constructor args (tau1, tau2, tau3, w1, w2, w3, rho0, rho1, nu, c1,
    c2, c3, beta_min=0.2, beta_max=1.5, sigma_min=0.2, sigma_max=1.5) are
    no longer accepted. Update the ClosedLoopGaussianScheduler(...) call in
    grpo_train.py to use the new argument names. All defaults are sensible
    for the user's Llama 3.2 3B + LoRA {q,k,v,o}_proj Countdown setup.
    """

    def __init__(
        self,
        num_tasks: int,
        *,
        # Reference trajectory (matches fixed E2H-G baseline)
        beta_ref: float = 0.50,
        sigma_ref: float = 0.50,
        # Initial state (defaults to the reference point)
        beta_t_init: Optional[float] = None,
        sigma_t_init: Optional[float] = None,
        # Bounds — tight around the reference for safety
        beta_min: float = 0.35,
        beta_max: float = 0.70,
        sigma_min: float = 0.40,
        sigma_max: float = 0.80,
        # Gains and smoothing
        eta_beta: float = 0.50,
        eta_sigma: float = 0.50,
        lambda_beta: float = 0.30,
        lambda_sigma: float = 0.30,
        # Per-bucket calibration (defaults for Llama 3.2 3B + full LoRA + Countdown)
        high_A: Optional[List[float]] = None,
        low_A: Optional[List[float]] = None,
        w_up: Optional[List[float]] = None,
        w_dn: Optional[List[float]] = None,
        # Sigma control
        c_fgt: float = 2.0,
        # Sampling floor
        min_prob: float = 0.01,
        # Reward stats (diagnostic only)
        reward_window_size: int = 100,
        reward_window_min_count: int = 32,
        stab_rho0: float = 0.70,
        stab_rho1: float = 0.40,
    ):
        if num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")

        # Store reference and bounds
        self.num_tasks = int(num_tasks)
        self.beta_ref = float(beta_ref)
        self.sigma_ref = float(sigma_ref)

        if not (0.0 < beta_min <= beta_ref <= beta_max):
            raise ValueError(
                f"Require 0 < beta_min <= beta_ref <= beta_max, "
                f"got ({beta_min}, {beta_ref}, {beta_max})"
            )
        if not (0.0 < sigma_min <= sigma_ref <= sigma_max):
            raise ValueError(
                f"Require 0 < sigma_min <= sigma_ref <= sigma_max, "
                f"got ({sigma_min}, {sigma_ref}, {sigma_max})"
            )

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        # Initial controller state
        self.beta_t = float(beta_t_init) if beta_t_init is not None else self.beta_ref
        self.sigma_t = float(sigma_t_init) if sigma_t_init is not None else self.sigma_ref
        self.beta_t = clip(self.beta_t, self.beta_min, self.beta_max)
        self.sigma_t = clip(self.sigma_t, self.sigma_min, self.sigma_max)

        # Gains and smoothing
        self.eta_beta = float(eta_beta)
        self.eta_sigma = float(eta_sigma)
        self.lambda_beta = float(lambda_beta)
        self.lambda_sigma = float(lambda_sigma)

        # Calibration vectors — defaults calibrated from the user's baselines
        if high_A is None:
            high_A = [0.60, 0.30, 0.15, 0.08][: self.num_tasks]
        if low_A is None:
            low_A = [0.30, 0.10, 0.04, 0.02][: self.num_tasks]
        if w_up is None:
            w_up = [0.25, 0.75, 1.00, 1.00][: self.num_tasks]
        if w_dn is None:
            w_dn = [1.00, 1.00, 0.75, 0.25][: self.num_tasks]

        for name, vec in [("high_A", high_A), ("low_A", low_A),
                          ("w_up", w_up), ("w_dn", w_dn)]:
            if len(vec) != self.num_tasks:
                raise ValueError(
                    f"{name} has length {len(vec)}, expected {self.num_tasks}"
                )
        for k in range(self.num_tasks):
            if not (low_A[k] < high_A[k]):
                raise ValueError(
                    f"Require low_A[{k}]={low_A[k]} < high_A[{k}]={high_A[k]}"
                )

        self.high_A = [float(x) for x in high_A]
        self.low_A = [float(x) for x in low_A]
        self.w_up = [float(x) for x in w_up]
        self.w_dn = [float(x) for x in w_dn]

        self.c_fgt = float(c_fgt)
        self.min_prob = float(min_prob)

        # Reward history (diagnostic)
        self.reward_window_size = int(reward_window_size)
        self.reward_window_min_count = int(reward_window_min_count)
        self.stab_rho0 = float(stab_rho0)
        self.stab_rho1 = float(stab_rho1)
        self.recent_rewards: deque[float] = deque(maxlen=self.reward_window_size)

        # Running best probe accuracy per bucket, for forgetting computation.
        self.best_probe_acc: Dict[int, float] = {k: 0.0 for k in range(self.num_tasks)}

    # ------------------------------------------------------------------
    # Public interface used by RewardTracker (unchanged)
    # ------------------------------------------------------------------

    def update_reward_stats(self, rewards: List[float]) -> None:
        for r in rewards:
            self.recent_rewards.append(float(r))

    def reward_mean(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def reward_variance(self) -> float:
        n = len(self.recent_rewards)
        if n <= 1:
            return 0.0
        mu = self.reward_mean()
        return sum((r - mu) ** 2 for r in self.recent_rewards) / n

    def has_enough_reward_history(self) -> bool:
        return len(self.recent_rewards) >= self.reward_window_min_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _curriculum_center(self, step_t: int, total_steps: int, beta: float) -> float:
        """x_t = (t/(T-1))^beta * (K-1)."""
        if total_steps <= 1:
            return float(self.num_tasks - 1)
        t = clip(step_t, 0, total_steps - 1)
        frac = t / float(total_steps - 1)
        return (frac ** beta) * float(self.num_tasks - 1)

    def _gaussian_probs(self, x: float, sigma: float) -> List[float]:
        """p(k) ∝ exp(-(k-x)^2 / (2 sigma^2)) + min_prob, normalized."""
        weights = [
            math.exp(-((k - x) ** 2) / (2.0 * sigma * sigma)) + self.min_prob
            for k in range(self.num_tasks)
        ]
        total = sum(weights)
        if total <= 0.0:
            return [1.0 / self.num_tasks] * self.num_tasks
        return [w / total for w in weights]

    def forgetting(self, bucket_id: int, current_acc: float) -> float:
        prev_best = self.best_probe_acc.get(bucket_id, 0.0)
        return positive_part(prev_best - float(current_acc))

    def _soft_stability(self, A: List[float]) -> float:
        """Diagnostic in [0, 1], not used in the control law."""
        if not self.has_enough_reward_history():
            return 0.0
        a0 = min(1.0, A[0] / max(self.stab_rho0, 1e-9))
        a1 = min(1.0, A[1] / max(self.stab_rho1, 1e-9)) if len(A) > 1 else 0.0
        return a0 * a1

    # ------------------------------------------------------------------
    # Public interface used by TaskSampler
    # ------------------------------------------------------------------

    def probs_for_step(self, step_t: int, total_steps: int) -> List[float]:
        """Return sampling probabilities at optimizer-step t using current
        controller state (beta_t, sigma_t)."""
        x = self._curriculum_center(step_t, total_steps, self.beta_t)
        return self._gaussian_probs(x, self.sigma_t)

    def baseline_probs_for_step(self, step_t: int, total_steps: int) -> List[float]:
        """What fixed E2H-G would have sampled at the same step. Purely
        diagnostic — use this to log how much the adaptive controller has
        actually deviated from the baseline trajectory."""
        x = self._curriculum_center(step_t, total_steps, self.beta_ref)
        return self._gaussian_probs(x, self.sigma_ref)

    # ------------------------------------------------------------------
    # Controller update, called from the probe callback
    # ------------------------------------------------------------------

    def update_from_probe_metrics(
        self,
        probe_acc_by_bucket: Dict[int, float],
        global_step: int,
    ) -> Dict[str, float]:
        """Update beta_t and sigma_t from a new batch of probe accuracies.

        Returns a dict with all fields used by the existing CSV writer in
        ClosedLoopProbeCallback, plus new diagnostic fields (speed_up,
        slow_down, baseline_beta, baseline_sigma).
        """
        # Collect A vector in bucket order 0..K-1
        A = [float(probe_acc_by_bucket.get(k, 0.0)) for k in range(self.num_tasks)]

        # Forgetting only on the first two buckets (matches previous schema).
        F0 = self.forgetting(0, A[0])
        F1 = self.forgetting(1, A[1]) if self.num_tasks > 1 else 0.0

        reward_var = self.reward_variance()

        # ------- Compute u_beta from symmetric calibrated signals -------
        speed_up = sum(
            self.w_up[k] * positive_part(A[k] - self.high_A[k])
            for k in range(self.num_tasks)
        )
        slow_down = sum(
            self.w_dn[k] * positive_part(self.low_A[k] - A[k])
            for k in range(self.num_tasks)
        )
        u_beta = speed_up - slow_down

        # ------- beta update (anchored to beta_ref) -------
        beta_old = self.beta_t
        beta_target = clip(
            self.beta_ref - self.eta_beta * u_beta,
            self.beta_min,
            self.beta_max,
        )
        beta_new = (1.0 - self.lambda_beta) * self.beta_t + self.lambda_beta * beta_target
        beta_new = clip(beta_new, self.beta_min, self.beta_max)

        # ------- u_sigma (positive-only signal: widen on forgetting) -------
        u_sigma = self.c_fgt * max(F0, F1)

        # ------- sigma update (anchored to sigma_ref) -------
        sigma_old = self.sigma_t
        sigma_target = clip(
            self.sigma_ref + self.eta_sigma * u_sigma,
            self.sigma_min,
            self.sigma_max,
        )
        sigma_new = (1.0 - self.lambda_sigma) * self.sigma_t + self.lambda_sigma * sigma_target
        sigma_new = clip(sigma_new, self.sigma_min, self.sigma_max)

        # ------- Commit state -------
        self.beta_t = beta_new
        self.sigma_t = sigma_new
        for k in range(self.num_tasks):
            self.best_probe_acc[k] = max(self.best_probe_acc.get(k, 0.0), A[k])

        # ------- Soft stability (diagnostic only) -------
        stability = self._soft_stability(A)

        # Pad A/F to length 4 for backward compatibility with the CSV schema.
        A_padded = A + [0.0] * max(0, 4 - len(A))

        log = ControllerUpdateLog(
            global_step=int(global_step),
            A0=A_padded[0],
            A1=A_padded[1],
            A2=A_padded[2],
            A3=A_padded[3],
            F0=F0,
            F1=F1,
            reward_mean=self.reward_mean(),
            reward_var=reward_var,
            u_beta=u_beta,
            u_sigma=u_sigma,
            beta_old=beta_old,
            beta_target=beta_target,
            beta_new=beta_new,
            sigma_old=sigma_old,
            sigma_target=sigma_target,
            sigma_new=sigma_new,
            stability=stability,
            speed_up=speed_up,
            slow_down=slow_down,
            baseline_beta=self.beta_ref,
            baseline_sigma=self.sigma_ref,
        )
        return log.__dict__