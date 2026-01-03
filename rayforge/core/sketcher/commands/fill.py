from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, List, Tuple, Optional

from .base import SketchChangeCommand
from ..sketch import Fill

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class AddFillCommand(SketchChangeCommand):
    """Command to add a Fill to a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        boundary: List[Tuple[int, bool]],
        name: str = _("Add Fill"),
    ):
        super().__init__(sketch, name)
        self.fill: Optional[Fill] = None
        self._boundary = boundary

    def _do_execute(self) -> None:
        if self.fill is None:
            self.fill = Fill(uid=str(uuid.uuid4()), boundary=self._boundary)
        self.sketch.fills.append(self.fill)

    def _do_undo(self) -> None:
        if self.fill and self.fill in self.sketch.fills:
            self.sketch.fills.remove(self.fill)


class RemoveFillCommand(SketchChangeCommand):
    """Command to remove a Fill from a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        fill: Fill,
        name: str = _("Remove Fill"),
    ):
        super().__init__(sketch, name)
        self.fill = fill

    def _do_execute(self) -> None:
        if self.fill in self.sketch.fills:
            self.sketch.fills.remove(self.fill)

    def _do_undo(self) -> None:
        self.sketch.fills.append(self.fill)
