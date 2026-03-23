from __future__ import annotations
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..constraints import HorizontalConstraint, VerticalConstraint
from ..entities import Line, Point
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
        self.add_cmd: Optional[AddItemsCommand] = None

    @staticmethod
    def calculate_geometry(
        rows: int,
        cols: int,
        origin: tuple[float, float],
        cell_width: float,
        cell_height: float,
        construction: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates points, entities, and constraints for a grid.

        Args:
            rows: Number of cell rows (horizontal bands)
            cols: Number of cell columns (vertical bands)
            origin: Bottom-left corner of the grid
            cell_width: Width of each cell
            cell_height: Height of each cell
            construction: Whether to create as construction geometry

        Returns:
            Dict with 'points', 'entities', and 'constraints' keys, or None
            if invalid.
        """
        if rows < 1 or cols < 1:
            return None
        if cell_width <= 0 or cell_height <= 0:
            return None

        rows = rows + 1
        cols = cols + 1
        ox, oy = origin
        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        points: List[Point] = []
        point_ids: List[int] = []

        for row in range(rows):
            for col in range(cols):
                pid = next_temp_id()
                x = ox + col * cell_width
                y = oy + row * cell_height
                points.append(Point(pid, x, y))
                point_ids.append(pid)

        def get_point_id(row: int, col: int) -> int:
            return point_ids[row * cols + col]

        entities: List[Line] = []
        constraints: List[Any] = []

        for row in range(rows):
            for col in range(cols - 1):
                p1_id = get_point_id(row, col)
                p2_id = get_point_id(row, col + 1)
                entities.append(
                    Line(
                        next_temp_id(), p1_id, p2_id, construction=construction
                    )
                )
                constraints.append(HorizontalConstraint(p1_id, p2_id))

        for col in range(cols):
            for row in range(rows - 1):
                p1_id = get_point_id(row, col)
                p2_id = get_point_id(row + 1, col)
                entities.append(
                    Line(
                        next_temp_id(), p1_id, p2_id, construction=construction
                    )
                )
                constraints.append(VerticalConstraint(p1_id, p2_id))

        return {
            "points": points,
            "entities": entities,
            "constraints": constraints,
        }

    def _do_execute(self) -> None:
        if self.add_cmd:
            return self.add_cmd._do_execute()

        result = self.calculate_geometry(
            self.rows,
            self.cols,
            self.origin,
            self.cell_width,
            self.cell_height,
            self.construction,
        )
        if not result:
            return

        self.add_cmd = AddItemsCommand(
            self.sketch,
            "",
            points=result["points"],
            entities=result["entities"],
            constraints=result["constraints"],
        )
        self.add_cmd._do_execute()

    def _do_undo(self) -> None:
        if self.add_cmd:
            self.add_cmd._do_undo()
