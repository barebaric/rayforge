from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint
from raygeo.sketch.entities.line import Line

from .base import SketchChangeCommand
from .items import AddItemsCommand

if TYPE_CHECKING:
    from ..sketch import Sketch


class GridCommand(SketchChangeCommand):
    """A command to create a homogeneous grid of construction lines."""

    def __init__(
        self,
        sketch: Sketch,
        rows: int,
        cols: int,
        origin: tuple[float, float] = (0, 0),
        cell_width: float = 10.0,
        cell_height: float = 10.0,
        construction: bool = True,
    ):
        super().__init__(sketch, _("Add Grid"))
        self.rows = rows
        self.cols = cols
        self.origin = origin
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.construction = construction
        self.add_cmd: AddItemsCommand | None = None
        self._grid_pids: list[int] = []

    @staticmethod
    def grid_point_ids(
        rows: int,
        cols: int,
        origin: tuple[float, float],
        cell_width: float,
        cell_height: float,
        add_point,
    ) -> list[int] | None:
        """Allocates the grid's point ids through ``add_point``,
        row-major from the bottom-left corner. Returns None for
        invalid parameters."""
        if rows < 1 or cols < 1:
            return None
        if cell_width <= 0 or cell_height <= 0:
            return None

        rows = rows + 1
        cols = cols + 1
        ox, oy = origin

        point_ids: list[int] = []
        for row in range(rows):
            for col in range(cols):
                x = ox + col * cell_width
                y = oy + row * cell_height
                point_ids.append(add_point(x, y))
        return point_ids

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        add_cmd = AddItemsCommand(self.sketch, "")
        point_ids = self.grid_point_ids(
            self.rows,
            self.cols,
            self.origin,
            self.cell_width,
            self.cell_height,
            add_cmd.add_point,
        )
        if point_ids is None:
            return

        rows = self.rows + 1
        cols = self.cols + 1

        def get_point_id(row: int, col: int) -> int:
            return point_ids[row * cols + col]

        for row in range(rows):
            for col in range(cols - 1):
                p1_id = get_point_id(row, col)
                p2_id = get_point_id(row, col + 1)
                add_cmd.add_entity(
                    Line(p1_id, p2_id, construction=self.construction)
                )
                add_cmd.add_constraint(HorizontalConstraint(p1_id, p2_id))

        for col in range(cols):
            for row in range(rows - 1):
                p1_id = get_point_id(row, col)
                p2_id = get_point_id(row + 1, col)
                add_cmd.add_entity(
                    Line(p1_id, p2_id, construction=self.construction)
                )
                add_cmd.add_constraint(VerticalConstraint(p1_id, p2_id))

        self.add_cmd = add_cmd
        self._grid_pids = point_ids

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
