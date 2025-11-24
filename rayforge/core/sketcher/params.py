import math
from typing import Dict, Any


class ParameterContext:
    """
    Manages named parameters and evaluates string expressions
    (e.g. 'width / 2').
    """

    def __init__(self) -> None:
        self._expressions: Dict[str, str] = {}
        self._cache: Dict[str, float] = {}
        self._dirty: bool = False

        # Safe math context
        self._math_context = {
            k: v for k, v in vars(math).items() if not k.startswith("_")
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the parameter context to a dictionary."""
        return {"expressions": self._expressions}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterContext":
        """Deserializes a dictionary into a ParameterContext instance."""
        new_context = cls()
        new_context._expressions = data.get("expressions", {})
        new_context._dirty = True  # Force re-evaluation on next get
        return new_context

    def set(self, name: str, value: float | str) -> None:
        """Sets a parameter. Can be a float or a math string."""
        self._expressions[name] = str(value)
        self._dirty = True

    def get(self, name: str, default: float = 0.0) -> float:
        """Gets the evaluated value of a parameter."""
        if self._dirty:
            self.evaluate_all()
        return self._cache.get(name, default)

    def evaluate(self, expression: str | float) -> float:
        """Evaluates an arbitrary expression string using current context."""
        if isinstance(expression, (int, float)):
            return float(expression)

        if self._dirty:
            self.evaluate_all()

        # Check if it's just a variable name
        if expression in self._cache:
            return self._cache[expression]

        try:
            # Merge math context with current variable values
            ctx = self._math_context.copy()
            ctx.update(self._cache)
            return float(eval(str(expression), {"__builtins__": None}, ctx))
        except Exception:
            return 0.0

    def evaluate_all(self) -> None:
        """
        Iteratively resolves dependencies.
        Simple multi-pass solver to handle out-of-order definitions.
        """
        self._cache.clear()
        # Max iterations equal to number of params to prevent infinite loops
        max_passes = len(self._expressions) + 1

        for _ in range(max_passes):
            progress = False
            ctx = self._math_context.copy()
            ctx.update(self._cache)

            for name, expr in self._expressions.items():
                if name in self._cache:
                    continue

                try:
                    val = float(eval(expr, {"__builtins__": None}, ctx))
                    self._cache[name] = val
                    ctx[name] = val
                    progress = True
                except (NameError, TypeError, SyntaxError):
                    # Dependency missing, try next pass
                    pass

            if not progress:
                break

        self._dirty = False
