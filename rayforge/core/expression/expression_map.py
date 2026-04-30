import re
import logging
from typing import Any, Dict, Optional

from .evaluator import MATH_CONTEXT

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


class ExpressionMap:
    """
    Evaluates ``{expression}`` placeholders in template strings using
    a sandboxed eval namespace.

    Uses the same ``MATH_CONTEXT`` as ``safe_evaluate()``, but returns
    whatever type the expression produces (no float coercion).

    Supports Python format specs: ``{value:04d}``, ``{width:.1f}``, etc.

    Unlike ``str.format_map()``, this uses regex-based substitution
    so expressions can contain attribute access
    (e.g. ``{date.today()}``).

    Usage::

        values = {"width": 50.0, "height": 30.0}
        tpl = "Part {width:.1f}x{height:.1f}"
        resolved = ExpressionMap(values).format(tpl)
        # "Part 50.0x30.0"
    """

    def __init__(self, values: Optional[Dict[str, Any]] = None):
        self._namespace: Dict[str, Any] = MATH_CONTEXT.copy()
        if values:
            self._namespace.update(values)
        self._namespace["__builtins__"] = {}

    def format(self, template: str) -> str:
        """Resolve all ``{expression}`` placeholders in *template*."""

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            expr, _, fmt = key.partition(":")
            try:
                result = eval(expr.strip(), self._namespace)
            except Exception as e:
                logger.debug(f"Failed to evaluate expression '{expr}': {e}")
                return match.group(0)
            if fmt:
                try:
                    return format(result, fmt)
                except (ValueError, TypeError):
                    return str(result)
            return str(result)

        return _PLACEHOLDER_RE.sub(_replace, template)
