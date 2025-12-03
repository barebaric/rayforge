import math
import pytest
from rayforge.core.expression import (
    context,
    errors,
    validator,
)


@pytest.fixture
def sample_context():
    """Provides a pre-populated ExpressionContext for validation tests."""
    variables = {"width": float, "count": int, "name": str}
    functions = {"sin": math.sin}
    return context.ExpressionContext(variables=variables, functions=functions)


@pytest.fixture
def v() -> validator.ExpressionValidator:
    """Provides an instance of the validator."""
    return validator.ExpressionValidator()


# --- Success Cases ---


def test_validate_valid_expression(v, sample_context):
    """Tests a valid expression with known variables."""
    result = v.validate("width / 2", sample_context)
    assert result.is_valid is True
    assert result.error_info is None


def test_validate_valid_function_call(v, sample_context):
    """Tests a valid expression with a known function."""
    result = v.validate("sin(width * 0.5)", sample_context)
    assert result.is_valid is True


def test_validate_empty_and_whitespace_string(v, sample_context):
    """Empty or whitespace-only strings should be considered valid."""
    assert v.validate("", sample_context).is_valid is True
    assert v.validate("   ", sample_context).is_valid is True


# --- Failure Cases ---


def test_validate_syntax_error(v, sample_context):
    """Tests an expression with invalid syntax."""
    result = v.validate("10 + * 5", sample_context)
    assert result.is_valid is False
    assert isinstance(result.error_info, errors.SyntaxErrorInfo)
    assert "syntax" in result.error_info.get_message().lower()


def test_validate_unknown_variable(v, sample_context):
    """Tests an expression with an undefined variable."""
    result = v.validate("width + height", sample_context)
    assert result.is_valid is False
    assert isinstance(result.error_info, errors.UnknownVariableInfo)
    assert result.error_info.name == "height"
    assert "height" in result.error_info.get_message()


def test_validate_type_mismatch_str_plus_int(v, sample_context):
    """Tests adding a known string variable to a number."""
    result = v.validate("name + 5", sample_context)
    assert result.is_valid is False
    assert isinstance(result.error_info, errors.TypeMismatchInfo)
    assert result.error_info.operator == "+"
    assert result.error_info.left_type == "str"
    assert result.error_info.right_type == "int"


def test_validate_type_mismatch_int_plus_str_literal(v, sample_context):
    """Tests adding a number to a string literal."""
    result = v.validate("count + 'abc'", sample_context)
    assert result.is_valid is False
    assert isinstance(result.error_info, errors.TypeMismatchInfo)
    assert result.error_info.left_type == "int"
    assert result.error_info.right_type == "str"
