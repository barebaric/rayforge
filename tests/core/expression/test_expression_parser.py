import ast
import pytest
from rayforge.core.expression import parser


@pytest.fixture
def p() -> parser.ExpressionParser:
    """Provides an instance of the parser."""
    return parser.ExpressionParser()


def test_parse_valid_expression(p):
    """Tests that a valid expression returns an AST object."""
    ast_node = p.parse("width / 2")
    assert ast_node is not None
    assert isinstance(ast_node, ast.AST)


def test_parse_invalid_expression(p):
    """Tests that an invalid expression returns None."""
    assert p.parse("width /") is None
    assert p.parse("10 + * 5") is None


def test_parse_empty_string(p):
    """Tests that an empty string returns None."""
    assert p.parse("") is None
    assert p.parse("   ") is None


def test_get_used_variables_simple(p):
    """Tests variable extraction from a simple expression."""
    ast_node = p.parse("a + b - 10")
    variables = p.get_used_variables(ast_node)
    assert variables == {"a", "b"}


def test_get_used_variables_with_function_call(p):
    """
    Tests extraction from an expression with a function call.
    The function name is correctly identified as a used name.
    """
    ast_node = p.parse("sin(radius * 2)")
    variables = p.get_used_variables(ast_node)
    assert variables == {"sin", "radius"}


def test_get_used_variables_with_repetition(p):
    """Tests that repeated variables appear only once in the resulting set."""
    ast_node = p.parse("x + x*x - 5/x")
    variables = p.get_used_variables(ast_node)
    assert variables == {"x"}


def test_get_used_variables_no_variables(p):
    """Tests an expression with only constants."""
    ast_node = p.parse("100 / (5 - 3)")
    variables = p.get_used_variables(ast_node)
    assert variables == set()
