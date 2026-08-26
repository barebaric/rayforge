import ast
import logging
import math
from typing import Any

from rayforge.core.expression import safe_evaluate

logger = logging.getLogger(__name__)


def _unresolved_names(expression: str, available: set[str]) -> set[str]:
    """Returns the set of names referenced by *expression* that are not
    in *available* (the set of currently-resolved variables and math
    functions). Used to tell forward references (retry on the next pass)
    apart from genuine errors (typos, bad syntax)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return set()

    class NameVisitor(ast.NodeVisitor):
        def __init__(self):
            self.names: set[str] = set()

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                self.names.add(node.id)
            self.generic_visit(node)

    visitor = NameVisitor()
    visitor.visit(tree)
    return visitor.names - available


class ParameterContext:
    """
    Manages named parameters and evaluates string expressions
    (e.g. 'width / 2').
    """

    def __init__(self) -> None:
        self._expressions: dict[str, str] = {}
        self._cache: dict[str, Any] = {}
        self._dirty: bool = False

        # Safe math context
        self._math_context = {
            k: v for k, v in vars(math).items() if not k.startswith("_")
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializes the parameter context to a dictionary."""
        return {"expressions": self._expressions.copy()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterContext":
        """Deserializes a dictionary into a ParameterContext instance."""
        new_context = cls()
        new_context._expressions = data.get("expressions", {})
        new_context._dirty = True  # Force re-evaluation on next get
        return new_context

    def set(self, name: str, value: float | str) -> None:
        """Sets a parameter. Can be a float or a math string."""
        self._expressions[name] = str(value)
        self._dirty = True

    def get(self, name: str, default: Any = 0.0) -> Any:
        """Gets the evaluated value of a parameter."""
        if self._dirty:
            self.evaluate_all()
        return self._cache.get(name, default)

    def get_all_values(self) -> dict[str, Any]:
        """Evaluates all expressions and returns a dictionary of all values."""
        if self._dirty:
            self.evaluate_all()
        return self._cache.copy()

    def evaluate(self, expression: str | float) -> Any:
        """Evaluates an arbitrary expression string using current context.

        Raises:
            ValueError: If the expression is invalid (e.g. a typo like
                'widht/2'). Numeric inputs never raise.
        """
        if isinstance(expression, (int, float)):
            return float(expression)

        if self._dirty:
            self.evaluate_all()

        # Check if it's just a variable name
        if expression in self._cache:
            return self._cache[expression]

        # Merge math context with current variable values
        ctx = self._math_context.copy()
        if self._cache:
            ctx.update(self._cache)

        return safe_evaluate(str(expression), ctx)

    def evaluate_all(
        self, initial_values: dict[str, Any] | None = None
    ) -> None:
        """
        Iteratively resolves dependencies.
        Simple multi-pass solver to handle out-of-order definitions.

        Args:
            initial_values: An optional dictionary of pre-set values to seed
                            the evaluation cache with. These have the highest
                            precedence.
        """
        self._cache.clear()
        if initial_values:
            self._cache.update(initial_values)

        # Max iterations equal to number of params to prevent infinite loops
        max_passes = len(self._expressions) + 1

        for _ in range(max_passes):
            progress = False
            # Always start with a fresh context for each pass
            ctx = self._math_context.copy()
            # The cache may already contain initial_values
            if self._cache:
                ctx.update(self._cache)

            for name, expr in self._expressions.items():
                if name in self._cache:
                    continue

                eval_ctx = self._math_context.copy()
                eval_ctx.update(self._cache)
                # Skip forward references: a name that is not yet resolved
                # may be defined later in evaluation order, so retry it on
                # the next pass rather than logging a spurious error.
                available = set(eval_ctx)
                missing = _unresolved_names(expr, available)
                if missing:
                    continue

                try:
                    val = safe_evaluate(expr, eval_ctx)
                    self._cache[name] = val
                    progress = True
                except (ValueError, SyntaxError, TypeError) as e:
                    # A real error in the expression (typo, bad syntax):
                    # log it so typos are not silently hidden, and leave
                    # the value unresolved (defaults to 0.0 on get()).
                    logger.warning(
                        "Parameter %r expression %r failed: %s",
                        name,
                        expr,
                        e,
                    )

            if not progress:
                break

        # After all passes, any expression still unresolved references a
        # name that is neither a known variable nor a math function: this
        # is a typo or a genuinely missing dependency. Log it so the user
        # can find it instead of silently getting 0.0 (issue 6).
        for name, expr in self._expressions.items():
            if name in self._cache:
                continue
            eval_ctx = self._math_context.copy()
            eval_ctx.update(self._cache)
            available = set(eval_ctx)
            missing = _unresolved_names(expr, available)
            if missing:
                logger.warning(
                    "Parameter %r expression %r references "
                    "unknown name(s): %s",
                    name,
                    expr,
                    ", ".join(sorted(missing)),
                )

        self._dirty = False
