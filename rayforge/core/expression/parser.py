import ast
from typing import Set, Optional


class ExpressionParser:
    """
    Parses an expression string into an Abstract Syntax Tree (AST) and
    extracts information like used variable names.
    """

    def parse(self, expression: str) -> Optional[ast.AST]:
        """
        Parses an expression string into an AST object.

        Args:
            expression: The string to parse.

        Returns:
            The root ast.AST node if parsing is successful, otherwise None.
        """
        if not expression:
            return None
        try:
            return ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            return None

    def get_used_variables(self, ast_node: ast.AST) -> Set[str]:
        """
        Traverses a parsed AST to find all names used as variables or
          functions.

        Args:
            ast_node: The root of the AST to traverse.

        Returns:
            A set of all unique names found in the expression.
        """

        class VariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.names = set()

            def visit_Name(self, node: ast.Name):
                self.names.add(node.id)
                self.generic_visit(node)

        visitor = VariableVisitor()
        visitor.visit(ast_node)
        return visitor.names
