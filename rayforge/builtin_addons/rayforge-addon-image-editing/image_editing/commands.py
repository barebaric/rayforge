from collections.abc import Sequence

from rayforge.core.image_edit import ImageEditStep
from rayforge.core.undo import ChangePropertyCommand, HistoryManager
from rayforge.core.workpiece import WorkPiece


def _toggled(
    edits: tuple[ImageEditStep, ...], edit: ImageEditStep, add: bool
) -> tuple[ImageEditStep, ...]:
    if add:
        return edits if edit in edits else (*edits, edit)
    return tuple(e for e in edits if e != edit)


def toggle_image_edit(
    history: HistoryManager,
    workpieces: Sequence[WorkPiece],
    edit: ImageEditStep,
    name: str,
) -> bool:
    """
    Removes ``edit`` when every workpiece already has it and appends it
    to the others otherwise, as one undo step. Returns True when the
    edit was added.
    """
    if not workpieces:
        return False
    add = any(edit not in wp.image_edits for wp in workpieces)
    with history.transaction(name) as t:
        for wp in workpieces:
            new_edits = _toggled(wp.image_edits, edit, add)
            if new_edits == wp.image_edits:
                continue
            t.execute(
                ChangePropertyCommand(
                    target=wp,
                    property_name="image_edits",
                    new_value=new_edits,
                    old_value=wp.image_edits,
                    name=name,
                )
            )
    return add
