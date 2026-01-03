from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class ModifyConstraintCommand(SketchChangeCommand):
    """
    Command to modify the value or expression of a constraint.
    """

    def __init__(
        self,
        sketch: "Sketch",
        constraint: "Constraint",
        new_value: float,
        new_expression: Optional[str] = None,
        name: str = _("Edit Constraint"),
    ):
        super().__init__(sketch, name)
        self.constraint = constraint
        self.new_value = float(new_value)
        self.new_expression = new_expression

        self.old_value = float(constraint.value)
        self.old_expression = getattr(constraint, "expression", None)

    def _do_execute(self) -> None:
        self.constraint.value = self.new_value
        self.constraint.expression = self.new_expression

    def _do_undo(self) -> None:
        self.constraint.value = self.old_value
        self.constraint.expression = self.old_expression
