import math
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Allowed functions and constants in expressions
MATH_CONTEXT = {
    k: v for k, v in math.__dict__.items() if not k.startswith("__")
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
    # Note: Context overrides math functions if names collide.
    namespace = MATH_CONTEXT.copy()
    namespace.update(context)

    # Add the restricted builtins to the same namespace
    namespace["__builtins__"] = {}

    try:
        # Use Python's eval, passing the unified namespace as globals.
        # Eval will use this for both globals and locals.
        result = eval(expr, namespace)
        return float(result)
    except Exception as e:
        logger.error(f"Failed to evaluate expression '{expression}': {e}")
        raise ValueError(f"Invalid expression: {e}")
