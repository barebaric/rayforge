from typing import Dict, Type, Callable, Optional


class ExpressionContext:
    """
    Represents the set of available variables and functions for an expression.

    This class acts as a symbol table, providing types and callables to the
    parser, validator, and UI components.
    """

    def __init__(
        self,
        variables: Optional[Dict[str, Type]] = None,
        functions: Optional[Dict[str, Callable]] = None,
    ):
        """
        Args:
            variables: A dictionary mapping variable names to their Python
              types.
            functions: A dictionary mapping function names to their callables.
        """
        self.variables: Dict[str, Type] = variables or {}
        self.functions: Dict[str, Callable] = functions or {}

    def is_variable(self, name: str) -> bool:
        """Checks if a name corresponds to a known variable."""
        return name in self.variables

    def is_function(self, name: str) -> bool:
        """Checks if a name corresponds to a known function."""
        return name in self.functions

    def get_variable_type(self, name: str) -> Optional[Type]:
        """Returns the type of a known variable."""
        return self.variables.get(name)
