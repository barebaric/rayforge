from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, List, Tuple, Optional
from gettext import gettext as _
from rayforge.image.structures import FillStyle
from rayforge.core.color import ColorRGBA
from ..entities.text_box import TextBoxEntity
from ..sketch import Fill, DEFAULT_FILL_COLOR
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..sketch import Sketch

logger = logging.getLogger(__name__)


class AddFillCommand(SketchChangeCommand):
    """Command to add a Fill to a sketch."""

    def __init__(
        self,
        sketch: "Sketch",
        boundary: List[Tuple[int, bool]],
        style: FillStyle = FillStyle.SOLID,
        color: ColorRGBA = DEFAULT_FILL_COLOR,
        gradient_stops: Optional[List[Tuple[float, ColorRGBA]]] = None,
        gradient_angle: float = 0.0,
        name: str = _("Add Fill"),
    ):
        super().__init__(sketch, name)
        self.fill: Optional[Fill] = None
        self._boundary = boundary
        self._style = style
        self._color = color
        self._gradient_stops = gradient_stops
        self._gradient_angle = gradient_angle

    def _do_execute(self) -> None:
        if self.fill is None:
            self.fill = Fill(
                uid=str(uuid.uuid4()),
                boundary=self._boundary,
                style=self._style,
                color=self._color,
                gradient_stops=self._gradient_stops,
                gradient_angle=self._gradient_angle,
            )
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


class SetTextFillCommand(SketchChangeCommand):
    """Command to set or toggle the fill color on a TextBoxEntity."""

    def __init__(
        self,
        sketch: "Sketch",
        entity_id: int,
        fill_color: Optional[ColorRGBA],
        name: str = _("Set Text Fill"),
    ):
        super().__init__(sketch, name)
        self.entity_id = entity_id
        self.fill_color = fill_color

    def _do_execute(self) -> None:
        entity = self.sketch.registry.get_entity(self.entity_id)
        if isinstance(entity, TextBoxEntity):
            entity.fill_color = self.fill_color

    def _do_undo(self) -> None:
        pass
