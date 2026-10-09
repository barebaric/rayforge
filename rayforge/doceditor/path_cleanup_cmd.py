import logging
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo import Geometry, Matrix

from ..core.item import DocItem
from ..core.path_cleanup import (
    close_open_contours,
    geometries_match,
    join_open_contours,
    remove_duplicate_contours,
)
from ..core.undo import ChangePropertyCommand, ListItemCommand
from ..core.workpiece import WorkPiece
from .split_cmd import ContourSplitStrategy

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)

DEFAULT_DUPLICATE_TOLERANCE_MM = 0.01

GeometryOperation = Callable[[Geometry, float], tuple[Geometry, int]]


def _mm_scale(workpiece: WorkPiece) -> tuple[float, float]:
    """
    The factors that turn the workpiece's normalized geometry into
    millimetres. A degenerate axis keeps a factor of 1 so the scaling can
    be undone.
    """
    width, height = workpiece.size
    return (
        width if width > 1e-9 else 1.0,
        height if height > 1e-9 else 1.0,
    )


def _is_editable(workpiece: WorkPiece) -> bool:
    """
    Workpieces generated from a geometry provider such as a sketch are
    edited at their source, so path clean-up leaves them alone.
    """
    if workpiece.geometry_provider_uid:
        return False
    boundaries = workpiece.boundaries
    return boundaries is not None and not boundaries.is_empty()


def _world_geometry(workpiece: WorkPiece) -> Geometry | None:
    boundaries = workpiece.boundaries
    if boundaries is None or boundaries.is_empty():
        return None
    geo = boundaries.copy()
    geo.transform(workpiece.get_world_transform())
    return geo


def _rects_overlap(a: Geometry, b: Geometry, tolerance: float) -> bool:
    return all(abs(va - vb) <= tolerance for va, vb in zip(a.rect(), b.rect()))


class PathCleanupCmd:
    """
    Clean-up operations on the vector paths of workpieces: closing,
    joining, removing duplicates and breaking apart. Tolerances are given
    in millimetres, independent of the workpiece's scale.
    """

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def close_paths(
        self, workpieces: list[WorkPiece], tolerance_mm: float
    ) -> int:
        """
        Closes open contours whose start and end are within the tolerance.

        Returns:
            The number of closed contours.
        """
        return self._apply(
            workpieces, close_open_contours, tolerance_mm, _("Close paths")
        )

    def join_paths(
        self, workpieces: list[WorkPiece], tolerance_mm: float
    ) -> int:
        """
        Joins open contours whose end points meet within the tolerance.

        Returns:
            The number of joins made.
        """
        return self._apply(
            workpieces, join_open_contours, tolerance_mm, _("Join paths")
        )

    def delete_duplicates(
        self,
        workpieces: list[WorkPiece],
        tolerance_mm: float = DEFAULT_DUPLICATE_TOLERANCE_MM,
    ) -> int:
        """
        Removes duplicate contours inside each workpiece, and removes
        workpieces that are an exact copy of an earlier one in the list
        lying on top of it.

        Returns:
            The number of removed contours and workpieces.
        """
        duplicates = self._find_duplicate_workpieces(workpieces, tolerance_mm)
        remaining = [wp for wp in workpieces if wp not in duplicates]
        changes = self._compute_changes(
            remaining, remove_duplicate_contours, tolerance_mm
        )
        if not duplicates and not changes:
            return 0

        name = _("Delete duplicates")
        removed = len(duplicates)
        with self._editor.history_manager.transaction(name) as t:
            for workpiece in duplicates:
                t.execute(
                    ListItemCommand(
                        owner_obj=workpiece.parent,
                        item=workpiece,
                        undo_command="add_child",
                        redo_command="remove_child",
                        name=name,
                    )
                )
            for workpiece, new_geo, count in changes:
                t.execute(self._boundaries_command(workpiece, new_geo, name))
                removed += count
        return removed

    def break_apart(self, workpieces: list[WorkPiece]) -> list[DocItem]:
        """
        Splits each workpiece into one workpiece per contour.

        Returns:
            The newly created workpieces.
        """
        return self._editor.split.split_items(
            workpieces, strategy=ContourSplitStrategy(), name=_("Break apart")
        )

    def _apply(
        self,
        workpieces: list[WorkPiece],
        operation: GeometryOperation,
        tolerance_mm: float,
        name: str,
    ) -> int:
        changes = self._compute_changes(workpieces, operation, tolerance_mm)
        if not changes:
            return 0
        total = 0
        with self._editor.history_manager.transaction(name) as t:
            for workpiece, new_geo, count in changes:
                t.execute(self._boundaries_command(workpiece, new_geo, name))
                total += count
        return total

    def _compute_changes(
        self,
        workpieces: list[WorkPiece],
        operation: GeometryOperation,
        tolerance_mm: float,
    ) -> list[tuple[WorkPiece, Geometry, int]]:
        """
        Runs the operation on each workpiece's geometry in millimetres and
        returns the normalized results of the workpieces it changed.
        """
        changes = []
        for workpiece in workpieces:
            if not _is_editable(workpiece):
                continue
            boundaries = workpiece.boundaries
            assert boundaries is not None
            sx, sy = _mm_scale(workpiece)
            geo_mm = boundaries.copy()
            geo_mm.transform(Matrix.scale(sx, sy))
            new_geo, count = operation(geo_mm, tolerance_mm)
            if count == 0:
                continue
            new_geo.transform(Matrix.scale(1.0 / sx, 1.0 / sy))
            changes.append((workpiece, new_geo, count))
        return changes

    def _boundaries_command(
        self, workpiece: WorkPiece, new_geo: Geometry, name: str
    ) -> ChangePropertyCommand:
        def _on_changed():
            workpiece.clear_render_cache()
            workpiece.updated.send(workpiece)

        return ChangePropertyCommand(
            target=workpiece,
            property_name="_edited_boundaries",
            new_value=new_geo,
            old_value=workpiece._edited_boundaries,
            on_change_callback=_on_changed,
            name=name,
        )

    def _find_duplicate_workpieces(
        self, workpieces: list[WorkPiece], tolerance_mm: float
    ) -> list[WorkPiece]:
        """
        Finds workpieces whose world-space geometry matches an earlier
        workpiece in the list.
        """
        kept: list[tuple[WorkPiece, Geometry]] = []
        duplicates: list[WorkPiece] = []
        for workpiece in workpieces:
            geo = _world_geometry(workpiece)
            if geo is None or workpiece.parent is None:
                continue
            if any(
                _rects_overlap(geo, other, tolerance_mm)
                and geometries_match(geo, other, tolerance_mm)
                for _wp, other in kept
            ):
                duplicates.append(workpiece)
            else:
                kept.append((workpiece, geo))
        return duplicates
