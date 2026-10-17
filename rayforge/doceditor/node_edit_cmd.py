"""
NodeEditCmd: undoable node edits on the vector paths of imported
workpieces (move, delete and insert nodes, line/curve, Bezier handles).

The geometry work happens in raygeo; this command writes the result to
the workpiece's edited boundaries, as segment deletion does. Workpieces
drawn by a geometry provider (sketches) are left alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo import Geometry

from ..core.undo import ChangePropertyCommand
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from .editor import DocEditor


class NodeEditCmd:
    """Node edits on the boundaries of imported workpieces."""

    def __init__(self, editor: DocEditor):
        self._editor = editor

    def _apply(
        self,
        workpiece: WorkPiece,
        name: str,
        edit: Callable[[Geometry], tuple[Geometry, int]],
    ) -> int | None:
        """Runs *edit* on the boundaries and commits the result."""
        if workpiece.geometry_provider_uid:
            return None
        boundaries = workpiece.boundaries
        if boundaries is None or boundaries.is_empty():
            return None
        new_geo, index = edit(boundaries)

        def _on_changed():
            workpiece.clear_render_cache()
            workpiece.updated.send(workpiece)

        self._editor.history_manager.execute(
            ChangePropertyCommand(
                target=workpiece,
                property_name="_edited_boundaries",
                new_value=new_geo,
                old_value=workpiece._edited_boundaries,
                on_change_callback=_on_changed,
                name=name,
                coalesce=False,
            )
        )
        return index

    def move_node(
        self, workpiece: WorkPiece, index: int, x: float, y: float
    ) -> int | None:
        """Moves the node at *index* to (x, y), in boundary space."""
        return self._apply(
            workpiece, _("Move node"), lambda g: g.move_node(index, x, y)
        )

    def delete_node(self, workpiece: WorkPiece, index: int) -> int | None:
        """Deletes the node at *index*, joining its neighbours."""
        return self._apply(
            workpiece, _("Delete node"), lambda g: g.delete_node(index)
        )

    def insert_node(
        self, workpiece: WorkPiece, index: int, t: float
    ) -> int | None:
        """Adds a node on the segment at *index*, at parameter *t*."""
        return self._apply(
            workpiece, _("Insert node"), lambda g: g.insert_node(index, t)
        )

    def set_segments_curve(
        self, workpiece: WorkPiece, indices: Iterable[int], curve: bool
    ) -> int | None:
        """Turns the segments at *indices* into curves or lines."""

        def edit(geo: Geometry) -> tuple[Geometry, int]:
            last = -1
            for index in sorted(indices, reverse=True):
                geo, last = geo.set_segment_curve(index, curve)
            return geo, last

        name = _("Make curve") if curve else _("Make line")
        return self._apply(workpiece, name, edit)

    def move_handle(
        self,
        workpiece: WorkPiece,
        index: int,
        handle: int,
        x: float,
        y: float,
    ) -> int | None:
        """Moves handle 1 or 2 of the Bezier at *index* to (x, y)."""
        return self._apply(
            workpiece,
            _("Move handle"),
            lambda g: (g.move_handle(index, handle, x, y), index),
        )
