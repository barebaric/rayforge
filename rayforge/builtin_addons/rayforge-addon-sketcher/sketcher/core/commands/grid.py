from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.sketch.constraints.horizontal import HorizontalConstraint
from raygeo.sketch.constraints.vertical import VerticalConstraint
from raygeo.sketch.entities.line import Line
from raygeo.sketch.entities.point import SketchPoint

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

    @staticmethod
    def calculate_geometry(
        rows: int,
        cols: int,
        origin: tuple[float, float],
        cell_width: float,
        cell_height: float,
        construction: bool = False,
    ) -> dict | None:
        """Pure geometry of the grid: points, line entities and
        axis constraints, with temporary sequential point ids. Returns
        None for invalid parameters."""
        if rows < 1 or cols < 1:
            return None
        if cell_width <= 0 or cell_height <= 0:
            return None

        nrows, ncols = rows + 1, cols + 1
        ox, oy = origin
        points: list[SketchPoint] = []
        for row in range(nrows):
            for col in range(ncols):
                points.append(
                    SketchPoint(
                        row * ncols + col,
                        ox + col * cell_width,
                        oy + row * cell_height,
                    )
                )

        def pid(row: int, col: int) -> int:
            return row * ncols + col

        entities: list[Line] = []
        constraints: list = []
        for row in range(nrows):
            for col in range(ncols - 1):
                a, b = pid(row, col), pid(row, col + 1)
                entities.append(Line(a, b, construction=construction))
                constraints.append(HorizontalConstraint(a, b))
        for col in range(ncols):
            for row in range(nrows - 1):
                a, b = pid(row, col), pid(row + 1, col)
                entities.append(Line(a, b, construction=construction))
                constraints.append(VerticalConstraint(a, b))

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._redo()

        geometry = self.calculate_geometry(
            self.rows,
            self.cols,
            self.origin,
            self.cell_width,
            self.cell_height,
            self.construction,
        )
        if geometry is None:
            return

        add_cmd = AddItemsCommand(self.sketch, "")
        id_map: dict[int, int] = {}
        for point in geometry["points"]:
            id_map[point.id] = add_cmd.add_point(point.x, point.y)
        for entity in geometry["entities"]:
            add_cmd.add_entity(
                Line(
                    id_map[entity.p1_idx],
                    id_map[entity.p2_idx],
                    construction=entity.construction,
                )
            )
        for constraint in geometry["constraints"]:
            if isinstance(constraint, HorizontalConstraint):
                add_cmd.add_constraint(
                    HorizontalConstraint(
                        id_map[constraint.p1], id_map[constraint.p2]
                    )
                )
            else:
                add_cmd.add_constraint(
                    VerticalConstraint(
                        id_map[constraint.p1], id_map[constraint.p2]
                    )
                )

        self.add_cmd = add_cmd
        self._grid_pids = list(id_map.values())

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
