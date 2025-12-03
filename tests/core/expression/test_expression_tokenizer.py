import pytest
from rayforge.core.expression import tokenizer


@pytest.fixture
def tk() -> tokenizer.ExpressionTokenizer:
    """Provides an instance of the tokenizer."""
    return tokenizer.ExpressionTokenizer()


def test_tokenize_empty_string(tk):
    """Tokenizing an empty string should return an empty list."""
    assert tk.tokenize("") == []
    assert tk.tokenize("   ") == []


def test_tokenize_simple_expression(tk):
    """Tests a basic expression with a variable, operator, and number."""
    expression = "width + 10"
    tokens = tk.tokenize(expression)

    assert len(tokens) == 3
    assert tokens[0] == tokenizer.Token(
        type=tokenizer.TokenType.NAME, value="width", start=0, end=5
    )
    assert tokens[1] == tokenizer.Token(
        type=tokenizer.TokenType.OPERATOR, value="+", start=6, end=7
    )
    assert tokens[2] == tokenizer.Token(
        type=tokenizer.TokenType.NUMBER, value="10", start=8, end=10
    )


def test_tokenize_function_call(tk):
    """Tests a more complex expression with a function call and parentheses."""
    expression = "sin(radius * 2)"
    tokens = tk.tokenize(expression)

    assert len(tokens) == 6
    assert tokens[0] == tokenizer.Token(
        type=tokenizer.TokenType.NAME, value="sin", start=0, end=3
    )
    assert tokens[1] == tokenizer.Token(
        type=tokenizer.TokenType.PARENTHESIS, value="(", start=3, end=4
    )
    assert tokens[2] == tokenizer.Token(
        type=tokenizer.TokenType.NAME, value="radius", start=4, end=10
    )
    assert tokens[3] == tokenizer.Token(
        type=tokenizer.TokenType.OPERATOR, value="*", start=11, end=12
    )
    assert tokens[4] == tokenizer.Token(
        type=tokenizer.TokenType.NUMBER, value="2", start=13, end=14
    )
    assert tokens[5] == tokenizer.Token(
        type=tokenizer.TokenType.PARENTHESIS, value=")", start=14, end=15
    )


def test_tokenize_all_types(tk):
    """Tests an expression containing one of each major token type."""
    expression = "func(1.5, 'text')"
    tokens = tk.tokenize(expression)
    token_types = [t.type for t in tokens]

    assert tokenizer.TokenType.NAME in token_types
    assert tokenizer.TokenType.PARENTHESIS in token_types
    assert tokenizer.TokenType.NUMBER in token_types
    assert tokenizer.TokenType.COMMA in token_types
    assert tokenizer.TokenType.STRING in token_types


def test_tokenize_invalid_syntax(tk):
    """
    Tests that the tokenizer handles syntax errors gracefully.
    It should not raise an exception, as the validator is responsible for
    reporting the formal error.
    """
    # An incomplete expression
    expression = "10 *"
    tokens = tk.tokenize(expression)
    # It should successfully tokenize what it can
    assert len(tokens) == 2
    assert tokens[0].value == "10"
    assert tokens[1].value == "*"

    # A completely invalid expression
    expression = "10..5"
    # Tokenizer might fail and return empty, which is acceptable.
    assert isinstance(tk.tokenize(expression), list)
