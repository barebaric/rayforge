import ast
from typing import Type

from .context import ExpressionContext
from .errors import (
    ValidationResult,
    SyntaxErrorInfo,
    UnknownVariableInfo,
    TypeMismatchInfo,
)
from .parser import ExpressionParser


class ExpressionValidator:
    """
    Performs syntax, semantic, and basic type checking on an expression string.
    """

    def __init__(self):
        self._parser = ExpressionParser()

    def validate(
        self, expression: str, context: ExpressionContext
    ) -> ValidationResult:
        """
        Validates an expression against a given context.

        Checks for:
        1. Valid Python syntax.
        2. References to undefined variables or functions.
        3. Basic type mismatches (e.g., adding a string to a number).

        Args:
            expression: The expression string to validate.
            context: The context containing available symbols.

        Returns:
            A ValidationResult object with the outcome.
        """
        if not expression.strip():
            return ValidationResult.success()

        # 1. Syntax Check
        try:
            # Use Python's built-in compile for a more detailed syntax error
            compile(expression, "<string>", "eval")
        except SyntaxError as e:
            return ValidationResult.failure(
                SyntaxErrorInfo(e.msg, e.offset or 0)
            )

        # Re-parse to get AST for further checks
        ast_node = self._parser.parse(expression)
        if not ast_node:
            # This case is unlikely if compile() succeeded, but is a safeguard.
            return ValidationResult.failure(
                SyntaxErrorInfo("Invalid expression", 0)
            )

        # 2. Unknown Variable Check
        used_names = self._parser.get_used_variables(ast_node)
        for name in used_names:
            if not context.is_variable(name) and not context.is_function(name):
                return ValidationResult.failure(UnknownVariableInfo(name))

        # 3. Type Mismatch Check
        try:
            type_checker = _TypeCheckVisitor(context)
            type_checker.visit(ast_node)
        except _TypeMismatchError as e:
            return ValidationResult.failure(
                TypeMismatchInfo(e.op_str, e.left, e.right)
            )

        return ValidationResult.success()


# --- Internal Helper for Type Checking ---


class _TypeMismatchError(TypeError):
    """Custom exception for type checking visitor."""

    def __init__(self, op_str: str, left: Type, right: Type):
        self.op_str = op_str
        self.left = left
        self.right = right
        super().__init__(f"Type mismatch: {left} {op_str} {right}")


class _TypeCheckVisitor(ast.NodeVisitor):
    """An AST visitor to perform basic type inference and checking."""

    # Map AST operators to their string representation
    _OP_MAP = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
        ast.Mod: "%",
    }

    def __init__(self, context: ExpressionContext):
        self.context = context

    def visit(self, node: ast.AST) -> Type:
        # Override visit to ensure we return a type
        result = super().visit(node)
        if not isinstance(result, type):
            # Default to float for complex/unhandled types like function calls
            return float
        return result

    def visit_Expression(self, node: ast.Expression) -> Type:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Type:
        return type(node.value)

    def visit_Name(self, node: ast.Name) -> Type:
        # Assumes unknown variable check has already passed
        var_type = self.context.get_variable_type(node.id)
        return var_type or float  # Default to float for functions

    def visit_BinOp(self, node: ast.BinOp) -> Type:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)

        # Allow any numeric operation
        numeric_types = (int, float, bool)
        is_left_numeric = left_type in numeric_types
        is_right_numeric = right_type in numeric_types

        if is_left_numeric and is_right_numeric:
            # Promote to float if one operand is a float
            return float if float in (left_type, right_type) else int

        # Forbid numeric operations with strings
        if (is_left_numeric and right_type is str) or (
            left_type is str and is_right_numeric
        ):
            op_str = self._OP_MAP.get(type(node.op), "?")
            raise _TypeMismatchError(op_str, left_type, right_type)

        # Fallback for other combinations
        return float

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Type:
        return self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> Type:
        # We assume all functions return a numeric (float) type for simplicity
        return float
