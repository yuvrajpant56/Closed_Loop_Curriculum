from typing import Any, Dict, List

from rewards.equation_checker import CountdownEquationChecker


class CountdownRewardFunction:
    """
    Strict, constraint-gated reward function for Countdown.

    Reward policy:
    - 0.0 if format is invalid
    - 0.0 if <answer> tag is missing
    - 0.0 if syntax is invalid
    - 0.0 if provided numbers are not used exactly once
    - 0.1 if structure is valid, numbers are valid, but target is wrong
    - 1.0 if structure is valid, numbers are valid, and target is correct

    This prevents the model from receiving positive reward for illegal
    expressions that invent numbers or omit required numbers.
    """

    def __init__(self, tolerance: float = 1e-6, allow_implicit_think_open: bool = True):
        self.checker = CountdownEquationChecker(tolerance=tolerance)
        self.allow_implicit_think_open = allow_implicit_think_open

    def score_single(
        self,
        completion_text: str,
        prompt_numbers: List[int],
        target: int,
    ) -> Dict[str, Any]:
        check = self.checker.check_completion(
            completion_text=completion_text,
            prompt_numbers=prompt_numbers,
            target=target,
            allow_implicit_think_open=self.allow_implicit_think_open,
        )

        reward_breakdown: Dict[str, float] = {
            "format_gate": 0.0,
            "answer_tag_gate": 0.0,
            "syntax_gate": 0.0,
            "numbers_gate": 0.0,
            "valid_but_wrong_target": 0.0,
            "correct_solution": 0.0,
        }

        # Gate 1: overall format must be valid
        if not check.get("format_valid", False):
            return {
                "reward": 0.0,
                "reward_breakdown": reward_breakdown,
                "checker_result": check,
            }
        reward_breakdown["format_gate"] = 1.0

        # Gate 2: <answer> tag must exist
        if not check.get("has_answer_tag", False):
            return {
                "reward": 0.0,
                "reward_breakdown": reward_breakdown,
                "checker_result": check,
            }
        reward_breakdown["answer_tag_gate"] = 1.0

        # Gate 3: expression must be syntactically valid
        if not check.get("syntax_valid", False):
            return {
                "reward": 0.0,
                "reward_breakdown": reward_breakdown,
                "checker_result": check,
            }
        reward_breakdown["syntax_gate"] = 1.0

        # Gate 4: must use exactly the provided numbers exactly once
        if not check.get("numbers_valid", False):
            return {
                "reward": 0.0,
                "reward_breakdown": reward_breakdown,
                "checker_result": check,
            }
        reward_breakdown["numbers_gate"] = 1.0

        # Now the output is a legal Countdown expression.
        # Give a small partial reward if legal but target is wrong.
        if not check.get("target_valid", False):
            reward_breakdown["valid_but_wrong_target"] = 0.1
            return {
                "reward": 0.1,
                "reward_breakdown": reward_breakdown,
                "checker_result": check,
            }

        # Full reward only for a fully correct solution
        reward_breakdown["correct_solution"] = 1.0
        return {
            "reward": 1.0,
            "reward_breakdown": reward_breakdown,
            "checker_result": check,
        }

    def score_batch(
        self,
        completions: List[str],
        prompt_numbers_batch: List[List[int]],
        targets: List[int],
    ) -> List[Dict[str, Any]]:
        if not (len(completions) == len(prompt_numbers_batch) == len(targets)):
            raise ValueError(
                "Batch lengths must match: "
                f"len(completions)={len(completions)}, "
                f"len(prompt_numbers_batch)={len(prompt_numbers_batch)}, "
                f"len(targets)={len(targets)}"
            )

        results: List[Dict[str, Any]] = []
        for idx, (completion_text, prompt_numbers, target) in enumerate(
            zip(completions, prompt_numbers_batch, targets)
        ):
            result = self.score_single(
                completion_text=completion_text,
                prompt_numbers=prompt_numbers,
                target=target,
            )
            result["batch_index"] = idx
            results.append(result)
        return results


def countdown_reward_fn(
    completions: List[str],
    targets: List[int],
    numbers_batch: List[List[int]],
    allow_implicit_think_open: bool = True,
    **kwargs,
) -> List[float]:
    """
    GRPO-friendly reward function.

    Returns a list of scalar rewards.

    Strict policy:
    - illegal number usage => 0.0
    - legal but wrong target => 0.1
    - legal and correct => 1.0
    """
    if not (len(completions) == len(targets) == len(numbers_batch)):
        raise ValueError(
            "Batch lengths must match: "
            f"len(completions)={len(completions)}, "
            f"len(targets)={len(targets)}, "
            f"len(numbers_batch)={len(numbers_batch)}"
        )

    reward_engine = CountdownRewardFunction(
        allow_implicit_think_open=allow_implicit_think_open
    )

    rewards: List[float] = []

    task_ids = kwargs.get("task_ids", None)
    _ = task_ids  # reserved for future use

    for idx, (completion_text, target_i, numbers_i) in enumerate(
        zip(completions, targets, numbers_batch)
    ):
        try:
            result = reward_engine.score_single(
                completion_text=completion_text,
                prompt_numbers=numbers_i,
                target=target_i,
            )
            rewards.append(float(result["reward"]))
        except Exception as e:
            print(f"[REWARD ERROR] idx={idx}, error={e}", flush=True)
            rewards.append(0.0)

    return rewards