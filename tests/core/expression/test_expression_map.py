from datetime import date
from rayforge.core.expression import ExpressionMap


def _fmt(template, values=None):
    return ExpressionMap(values).format(template)


def test_plain_text_passes_through():
    """Plain text without placeholders is returned unchanged."""
    assert _fmt("Hello World") == "Hello World"


def test_empty_string():
    """Empty template produces empty string."""
    assert _fmt("") == ""


def test_numeric_variable_substitution():
    """Numeric values are substituted and stringified."""
    result = _fmt(
        "{width}x{height}", {"width": 50.0, "height": 30.0}
    )
    assert result == "50.0x30.0"


def test_format_spec_applied():
    """Python format specs are applied to evaluated values."""
    assert _fmt("{count:04d}", {"count": 7}) == "0007"
    assert _fmt("{ratio:.2f}", {"ratio": 3.14159}) == "3.14"


def test_math_expression_evaluated():
    """Expressions using math functions are evaluated."""
    assert _fmt("{sqrt(width)}", {"width": 100.0}) == "10.0"


def test_arithmetic_in_placeholder():
    """Arithmetic expressions work inside placeholders."""
    assert _fmt(
        "{width + height}", {"width": 50.0, "height": 30.0}
    ) == "80.0"


def test_mixed_text_and_expressions():
    """Template mixing literal text with expressions."""
    assert _fmt(
        "Part {width:.1f}x{height:.1f}mm",
        {"width": 50.0, "height": 30.0},
    ) == "Part 50.0x30.0mm"


def test_unknown_expression_left_as_is():
    """Unresolvable expressions are left as literal braces."""
    assert _fmt("{unknown_var}") == "{unknown_var}"


def test_sandboxed_no_builtins():
    """Builtins like __import__ are not accessible."""
    assert "{__import__" in _fmt("{__import__('os')}")


def test_callable_in_values():
    """Callables in values dict can be invoked in expressions."""
    assert _fmt("{double(5)}", {"double": lambda x: x * 2}) == "10"


def test_none_values():
    """ExpressionMap works with no values dict."""
    assert _fmt("no placeholders") == "no placeholders"


def test_context_overrides_math():
    """User values take precedence over math context."""
    assert _fmt("{pi}", {"pi": 3.0}) == "3.0"


def test_attribute_access():
    """Expressions with attribute access work (regex-based)."""
    today = date.today()
    result = _fmt("{d.isoformat()}", {"d": today})
    assert result == today.isoformat()


def test_chained_attribute_access():
    """Chained attribute access and method calls work."""
    today = date.today()
    result = _fmt("{d.year}", {"d": today})
    assert result == str(today.year)
