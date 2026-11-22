"""
Undoable replacement of a workpiece's edited boundaries, shared by the
commands that edit vector paths: segment and contour deletion, path
cleanup, and node editing.
"""

from raygeo.geo import Geometry

from ..core.undo import ChangePropertyCommand
from ..core.workpiece import WorkPiece


def edited_boundaries_command(
    workpiece: WorkPiece, new_geo: Geometry, name: str
) -> ChangePropertyCommand:
    """
    Returns an undoable command that replaces the edited boundaries of
    *workpiece* with *new_geo*. Executing it clears the render caches
    and sends the updated signal. Every boundary edit is a discrete
    user action, so the command opts out of coalescing: two quick edits
    stay two undo steps.
    """

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
        coalesce=False,
    )
