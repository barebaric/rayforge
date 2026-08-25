from __future__ import annotations

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.core.undo.command import Command

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
        sketch: Sketch,
        constraint: Constraint,
        new_value: float,
        new_expression: str | None = None,
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

    def can_coalesce_with(self, next_command: Command) -> bool:
        return (
            isinstance(next_command, ModifyConstraintCommand)
            and self.constraint is next_command.constraint
        )

    def coalesce_with(self, next_command: Command) -> bool:
        if not self.can_coalesce_with(next_command):
            return False
        assert isinstance(next_command, ModifyConstraintCommand)
        self.new_value = next_command.new_value
        self.new_expression = next_command.new_expression
        self.timestamp = next_command.timestamp
        return True
