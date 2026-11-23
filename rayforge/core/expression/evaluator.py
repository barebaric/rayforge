import logging
from typing import Any

from raygeo.template import evaluate_number

logger = logging.getLogger(__name__)


def safe_evaluate(expression: str, context: dict[str, Any]) -> float:
    """
    Evaluates a mathematical expression string using a specific context
    (variable names) and the built-in math functions.

    The evaluator lives in raygeo's wasm-safe template module, so the
    same expression semantics are available to every host.

    Args:
        expression: The string to evaluate (e.g., "width / 2 + 5").
        context: A dictionary of variable names to values.

    Returns:
        float: The calculated value.

    Raises:
        ValueError: If evaluation fails or syntax is invalid.
    """
    if not expression:
        return 0.0

    try:
        return evaluate_number(expression.strip(), context)
    except Exception as e:  # noqa: BLE001 - arbitrary user expression
        logger.error(f"Failed to evaluate expression '{expression}': {e}")
        raise ValueError(f"Invalid expression: {e}")
