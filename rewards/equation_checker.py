import ast
import math
import operator
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class SafeExpressionEvaluator:
    """
    Safely evaluate arithmetic expressions containing only:
      - numbers
      - +, -, *, /
      - parentheses
      - unary + and -

    No eval() is used.
    """

    def eval_expr(self, expr: str) -> float:
        parsed = ast.parse(expr, mode="eval")
        return float(self._eval_node(parsed.body))

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant: {node.value}")
            return float(node.value)

        if isinstance(node, ast.Num):  # older Python compatibility
            return float(node.n)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if op_type is ast.Div and abs(right) < 1e-12:
                raise ZeroDivisionError("Division by zero in expression.")

            return float(_ALLOWED_BIN_OPS[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return float(_ALLOWED_UNARY_OPS[op_type](operand))

        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CountdownEquationChecker:
    """
    Validates model output for the Countdown task.

    It checks:
      1. response format (<think>...</think><answer>...</answer>)
      2. answer extraction
      3. arithmetic syntax
      4. number usage
      5. target correctness

    Important:
    If your prompt already ends with '<think>' and generation starts inside the
    think block, you can call check_completion(..., allow_implicit_think_open=True).
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.evaluator = SafeExpressionEvaluator()

    def normalize_completion_for_format_check(
        self,
        completion_text: str,
        allow_implicit_think_open: bool = False,
    ) -> str:
        """
        If generation starts inside an already-open <think> block from the prompt,
        we can prepend '<think>' for validation purposes.

        Example generated completion:
            '82 - 15 = 67</think><answer>82 - 15</answer>'

        becomes:
            '<think>82 - 15 = 67</think><answer>82 - 15</answer>'
        """
        text = completion_text.strip()

        if allow_implicit_think_open:
            has_think_open = "<think>" in text
            has_think_close = "</think>" in text
            if (not has_think_open) and has_think_close:
                text = "<think>" + text

        return text

    def validate_countdown_response_format(
        self,
        response: str,
        allow_implicit_think_open: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate the formatting of the response.

        Required structure:
            <think> ... </think><answer> ... </answer>

        Rules checked:
        - think tags exist
        - answer tags exist
        - closing tags occur after opening tags
        - answer starts after think closes
        """
        response = self.normalize_completion_for_format_check(
            response,
            allow_implicit_think_open=allow_implicit_think_open,
        ).strip()

        format_result: Dict[str, Any] = {
            "format_valid": False,
            "normalized_response": response,
            "has_think_open": "<think>" in response,
            "has_think_close": "</think>" in response,
            "has_answer_open": "<answer>" in response,
            "has_answer_close": "</answer>" in response,
            "think_open_idx": -1,
            "think_close_idx": -1,
            "answer_open_idx": -1,
            "answer_close_idx": -1,
            "error": None,
            "details": [],
        }

        if "<think>" not in response or "</think>" not in response:
            format_result["error"] = "Response does not contain complete <think> tags."
            format_result["details"].append("Missing <think> or </think>.")
            return format_result

        if "<answer>" not in response or "</answer>" not in response:
            format_result["error"] = "Response does not contain complete <answer> tags."
            format_result["details"].append("Missing <answer> or </answer>.")
            return format_result

        think_open = response.find("<think>")
        think_close = response.find("</think>")
        answer_open = response.find("<answer>")
        answer_close = response.find("</answer>")

        format_result["think_open_idx"] = think_open
        format_result["think_close_idx"] = think_close
        format_result["answer_open_idx"] = answer_open
        format_result["answer_close_idx"] = answer_close

        if think_close < think_open:
            format_result["error"] = "</think> appears before <think>."
            format_result["details"].append("Invalid think tag ordering.")
            return format_result

        if answer_close < answer_open:
            format_result["error"] = "</answer> appears before <answer>."
            format_result["details"].append("Invalid answer tag ordering.")
            return format_result

        if answer_open < think_close:
            format_result["error"] = "<answer> appears before </think>."
            format_result["details"].append("Answer block must come after think block.")
            return format_result

        format_result["format_valid"] = True
        format_result["details"].append("Response format is valid.")
        return format_result

    def extract_think_tag(self, text: str) -> Optional[str]:
        match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_answer_tag(self, text: str) -> Optional[str]:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def normalize_answer_expression(self, answer_text: str) -> str:
        """
        If model outputs:
            '82 - 15 = 67'
        keep only:
            '82 - 15'
        """
        expr = answer_text.strip()
        if "=" in expr:
            expr = expr.split("=", 1)[0].strip()
        return expr

    def extract_numbers_from_expression(self, expr: str) -> List[float]:
        raw_numbers = re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?", expr)
        return [float(x) for x in raw_numbers]

    def _is_integer_like(self, value: float) -> bool:
        return math.isfinite(value) and abs(value - round(value)) < self.tolerance

    def _number_usage_check(
        self,
        expr_numbers: List[float],
        prompt_numbers: List[int],
    ) -> Tuple[bool, str, Counter, Counter]:
        if not expr_numbers:
            return False, "No numbers found in expression.", Counter(), Counter(prompt_numbers)

        if not all(self._is_integer_like(x) for x in expr_numbers):
            return (
                False,
                "Expression uses non-integer numbers not allowed by the prompt.",
                Counter(expr_numbers),
                Counter(prompt_numbers),
            )

        expr_ints = [int(round(x)) for x in expr_numbers]
        expr_counter = Counter(expr_ints)
        prompt_counter = Counter(prompt_numbers)

        if expr_counter != prompt_counter:
            return (
                False,
                f"Number usage mismatch. Expression uses {dict(expr_counter)}, "
                f"but prompt requires {dict(prompt_counter)}.",
                expr_counter,
                prompt_counter,
            )

        return True, "Expression uses exactly the prompt numbers.", expr_counter, prompt_counter

    def check_expression(
        self,
        expression: str,
        prompt_numbers: List[int],
        target: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "expression": expression,
            "target": target,
            "prompt_numbers": prompt_numbers,
            "syntax_valid": False,
            "numbers_valid": False,
            "target_valid": False,
            "is_correct": False,
            "used_numbers": [],
            "value": None,
            "error": None,
            "details": [],
        }

        expr = self.normalize_answer_expression(expression)
        result["expression"] = expr

        try:
            used_numbers = self.extract_numbers_from_expression(expr)
            result["used_numbers"] = used_numbers

            numbers_valid, msg, _, _ = self._number_usage_check(used_numbers, prompt_numbers)
            result["numbers_valid"] = numbers_valid
            result["details"].append(msg)

            value = self.evaluator.eval_expr(expr)
            result["syntax_valid"] = True
            result["value"] = value

            if math.isfinite(value) and abs(value - float(target)) < self.tolerance:
                result["target_valid"] = True
                result["details"].append(f"Expression evaluates to target: {value}.")
            else:
                result["details"].append(f"Expression evaluates to {value}, not target {target}.")

            result["is_correct"] = (
                result["syntax_valid"]
                and result["numbers_valid"]
                and result["target_valid"]
            )

        except Exception as exc:
            result["error"] = str(exc)
            result["details"].append(f"Expression check failed: {exc}")

        return result

    def check_completion(
        self,
        completion_text: str,
        prompt_numbers: List[int],
        target: int,
        allow_implicit_think_open: bool = False,
    ) -> Dict[str, Any]:
        format_info = self.validate_countdown_response_format(
            completion_text,
            allow_implicit_think_open=allow_implicit_think_open,
        )

        normalized_response = format_info["normalized_response"]

        result: Dict[str, Any] = {
            "format_valid": format_info["format_valid"],
            "format_info": format_info,
            "has_think_tag": format_info["has_think_open"] and format_info["has_think_close"],
            "has_answer_tag": format_info["has_answer_open"] and format_info["has_answer_close"],
            "raw_think_text": None,
            "raw_answer_text": None,
            "expression": None,
            "syntax_valid": False,
            "numbers_valid": False,
            "target_valid": False,
            "is_correct": False,
            "used_numbers": [],
            "value": None,
            "error": None,
            "details": [],
        }

        if not format_info["format_valid"]:
            result["error"] = format_info["error"]
            result["details"].extend(format_info["details"])
            return result

        think_text = self.extract_think_tag(normalized_response)
        answer_text = self.extract_answer_tag(normalized_response)

        result["raw_think_text"] = think_text
        result["raw_answer_text"] = answer_text

        if answer_text is None:
            result["error"] = "Missing valid <answer>...</answer> after format normalization."
            result["details"].append(result["error"])
            return result

        expr_result = self.check_expression(answer_text, prompt_numbers, target)
        result.update(expr_result)
        result["details"] = format_info["details"] + expr_result["details"]
        return result