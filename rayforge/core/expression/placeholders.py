"""
Names used by ``{...}`` placeholders in text templates.

The scanning is implemented in raygeo's wasm-safe Rust template module;
these thin wrappers keep the expression package's public API stable and
usable by host code.
"""

from __future__ import annotations

from collections.abc import Iterable

from raygeo.template import names_in_expression as _names_in_expression
from raygeo.template import referenced_names as _referenced_names


def names_in_expression(expression: str) -> set[str]:
    """Names an expression reads, ignoring string literals."""
    return set(_names_in_expression(expression))


def referenced_template_names(templates: Iterable[str]) -> set[str]:
    """
    Collects the names used inside the ``{...}`` placeholders of
    *templates*. A format spec after the first colon is ignored.
    """
    names: set[str] = set()
    for template in templates:
        names |= set(_referenced_names(template or ""))
    return names
