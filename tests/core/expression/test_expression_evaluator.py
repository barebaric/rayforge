import math
import pytest
from rayforge.core.expression import evaluator


def test_evaluate_simple_arithmetic():
    """Tests basic math operations without context."""
    assert evaluator.safe_evaluate("10 + 5", {}) == 15.0
    assert evaluator.safe_evaluate("10 * (4 - 2)", {}) == 20.0


def test_evaluate_with_context_variables():
    """Tests evaluation using variables from the context."""
    context = {"width": 100.0, "height": 50}
    assert evaluator.safe_evaluate("width / 2", context) == 50.0
    assert evaluator.safe_evaluate("width + height", context) == 150.0


def test_evaluate_with_math_functions():
    """Tests that standard math functions are available."""
    assert evaluator.safe_evaluate("pi", {}) == pytest.approx(math.pi)
    assert evaluator.safe_evaluate("sin(pi / 2)", {}) == pytest.approx(1.0)
    assert evaluator.safe_evaluate("sqrt(16)", {}) == 4.0


def test_evaluate_empty_string():
    """An empty string should evaluate to 0.0."""
    assert evaluator.safe_evaluate("", {}) == 0.0


def test_evaluate_context_overrides_math():
    """Tests that context variables take precedence over built-in math."""
    context = {"pi": 3.0, "sin": 10}
    assert evaluator.safe_evaluate("pi", context) == 3.0
    assert evaluator.safe_evaluate("sin * 2", context) == 20.0


def test_evaluate_raises_value_error_on_failure():
    """Ensures that evaluation failures raise a `ValueError`."""
    # Syntax error
    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.safe_evaluate("10 + * 5", {})

    # Unknown variable
    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.safe_evaluate("width + 5", {})

    # Type error
    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.safe_evaluate("'a' + 5", {})


def test_evaluate_prevents_builtin_access():
    """
    Crucially, tests that `eval` is properly sandboxed and cannot access
    dangerous built-in functions.
    """
    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.safe_evaluate("__import__('os')", {})

    with pytest.raises(ValueError, match="Invalid expression"):
        evaluator.safe_evaluate("open('file.txt')", {})
