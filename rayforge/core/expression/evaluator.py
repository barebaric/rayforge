import math
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Allowed functions in expressions
_MATH_FUNCTIONS = {
    k: v
    for k, v in math.__dict__.items()
    if not k.startswith("__") and callable(v)
}


def safe_evaluate(expression: str, context: Dict[str, Any]) -> float:
    """
    Evaluates a mathematical expression string using a specific context
    (variable names) and standard math functions.

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

    # Clean whitespace
    expr = expression.strip()

    # Create the evaluation namespace
    # We combine the user parameters (context) with standard math functions.
    # Note: Context overrides math functions if names collide.
    namespace = _MATH_FUNCTIONS.copy()
    namespace.update(context)

    try:
        # Use Python's eval, but restricted to the specific namespace.
        # We explicitly set __builtins__ to None to prevent access to
        # dangerous system functions.
        result = eval(expr, {"__builtins__": None}, namespace)
        return float(result)
    except Exception as e:
        logger.error(f"Failed to evaluate expression '{expression}': {e}")
        raise ValueError(f"Invalid expression: {e}")
