import enum
import io
import token as py_token
import tokenize as py_tokenize
from typing import NamedTuple, List, Optional


class TokenType(enum.Enum):
    """Simplified token types for syntax highlighting and parsing."""

    UNKNOWN = 0
    NAME = 1
    NUMBER = 2
    STRING = 3
    OPERATOR = 4
    PARENTHESIS = 5
    COMMA = 6


class Token(NamedTuple):
    """Represents a single token with its type, value, and position."""

    type: TokenType
    value: str
    start: int
    end: int


class ExpressionTokenizer:
    """
    Breaks an expression string into a sequence of classified tokens for
    syntax highlighting.
    """

    _TYPE_MAP = {
        py_token.NAME: TokenType.NAME,
        py_token.NUMBER: TokenType.NUMBER,
        py_token.STRING: TokenType.STRING,
        py_token.OP: TokenType.OPERATOR,
    }

    def tokenize(self, expression: str) -> List[Token]:
        """
        Converts an expression string into a list of Token objects.

        Args:
            expression: The string to tokenize.

        Returns:
            A list of Tokens representing the expression.
        """
        if not expression.strip():
            return []

        tokens: List[Token] = []
        try:
            # The tokenize module expects a callable that returns strings.
            # io.StringIO provides this.
            source = io.StringIO(expression)
            tok_gen = py_tokenize.generate_tokens(source.readline)

            for tok in tok_gen:
                token_type = self._get_token_type(tok)
                if token_type:
                    # The tokenizer gives (line, col) tuples. For a single-line
                    # entry, we only care about the column.
                    start_col = tok.start[1]
                    end_col = tok.end[1]
                    tokens.append(
                        Token(token_type, tok.string, start_col, end_col)
                    )
        except (py_tokenize.TokenError, IndentationError, TypeError):
            # If tokenizing fails, it's a syntax error. Return an empty list;
            # the validator will catch the error more formally.
            return []

        return tokens

    def _get_token_type(
        self, tok: py_tokenize.TokenInfo
    ) -> Optional[TokenType]:
        """Maps a standard library token to our simplified TokenType."""
        # 1. Explicitly filter out all non-content tokens.
        if tok.type in (
            py_token.ENCODING,
            py_token.ENDMARKER,
            py_token.NEWLINE,
            py_token.NL,
            py_token.COMMENT,
            py_token.INDENT,
            py_token.DEDENT,
        ):
            return None

        # 2. Specific symbols take precedence over generic types.
        if tok.string in "()":
            return TokenType.PARENTHESIS
        if tok.string == ",":
            return TokenType.COMMA

        # 3. Fallback to generic type mapping for everything else.
        return self._TYPE_MAP.get(tok.type)
