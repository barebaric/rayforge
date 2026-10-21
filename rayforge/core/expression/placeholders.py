"""
Names used by ``{...}`` placeholders in text templates.

Templates embed expressions inside braces; a format spec may follow
the first colon, e.g. ``{counter:04d}``. These helpers collect the
variable names a set of template strings reads, so callers can ask
which templates depend on which variables without evaluating them.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def names_in_expression(expression: str) -> set[str]:
    """Names an expression reads, ignoring string literals."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return set(_IDENTIFIER_RE.findall(expression))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def referenced_template_names(templates: Iterable[str]) -> set[str]:
    """
    Collects the names used inside the ``{...}`` placeholders of
    *templates*. A format spec after the first colon is ignored.
    """
    names: set[str] = set()
    for template in templates:
        for match in _PLACEHOLDER_RE.finditer(template or ""):
            expression, _sep, _spec = match.group(1).partition(":")
            names |= names_in_expression(expression)
    return names
