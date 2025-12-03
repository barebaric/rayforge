import math
import pytest
from rayforge.core.expression import context


def sample_function():
    pass


@pytest.fixture
def sample_context():
    """Provides a pre-populated ExpressionContext for tests."""
    variables = {"width": float, "count": int}
    functions = {"sin": math.sin, "custom": sample_function}
    return context.ExpressionContext(variables=variables, functions=functions)


def test_context_initialization_empty():
    """Tests that an ExpressionContext can be created with no arguments."""
    ctx = context.ExpressionContext()
    assert ctx.variables == {}
    assert ctx.functions == {}


def test_context_initialization_with_data(sample_context):
    """Tests that an ExpressionContext is populated correctly on creation."""
    assert "width" in sample_context.variables
    assert sample_context.variables["width"] is float
    assert "count" in sample_context.variables
    assert sample_context.variables["count"] is int

    assert "sin" in sample_context.functions
    assert sample_context.functions["sin"] is math.sin
    assert "custom" in sample_context.functions
    assert sample_context.functions["custom"] is sample_function


def test_is_variable(sample_context):
    """Tests the is_variable method."""
    assert sample_context.is_variable("width") is True
    assert sample_context.is_variable("count") is True
    assert sample_context.is_variable("undefined_var") is False
    # A function name should not be identified as a variable
    assert sample_context.is_variable("sin") is False


def test_is_function(sample_context):
    """Tests the is_function method."""
    assert sample_context.is_function("sin") is True
    assert sample_context.is_function("custom") is True
    assert sample_context.is_function("undefined_func") is False
    # A variable name should not be identified as a function
    assert sample_context.is_function("width") is False


def test_get_variable_type(sample_context):
    """Tests the get_variable_type method."""
    assert sample_context.get_variable_type("width") is float
    assert sample_context.get_variable_type("count") is int
    assert sample_context.get_variable_type("undefined_var") is None
    assert sample_context.get_variable_type("sin") is None
