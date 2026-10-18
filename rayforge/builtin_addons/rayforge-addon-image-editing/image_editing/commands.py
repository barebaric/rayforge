from collections.abc import Mapping, Sequence

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


def replace_op_edits(
    edits: tuple[ImageEditStep, ...],
    replacements: Mapping[str, ImageEditStep | None],
) -> tuple[ImageEditStep, ...]:
    """
    Puts each replacement where the edit of the same op was, or appends
    it; ``None`` removes the op's edit. Other edits keep their order.
    """
    result = []
    placed = set()
    for edit in edits:
        if edit.op not in replacements:
            result.append(edit)
            continue
        new = replacements[edit.op]
        if new is not None and edit.op not in placed:
            result.append(new)
        placed.add(edit.op)
    for op, new in replacements.items():
        if new is not None and op not in placed:
            result.append(new)
    return tuple(result)


def set_op_edits(
    history: HistoryManager,
    workpieces: Sequence[WorkPiece],
    replacements: Mapping[str, ImageEditStep | None],
    name: str,
    old_edits: Mapping[str, tuple[ImageEditStep, ...]] | None = None,
) -> None:
    """
    Applies :func:`replace_op_edits` to every workpiece as one undo
    step. ``old_edits`` maps workpiece uids to the stacks to undo to,
    for callers that already showed the new edits as a live preview.
    """
    old_edits = old_edits or {}
    with history.transaction(name) as t:
        for wp in workpieces:
            old = old_edits.get(wp.uid, wp.image_edits)
            new = replace_op_edits(old, replacements)
            if new == old:
                wp.image_edits = old
                continue
            t.execute(
                ChangePropertyCommand(
                    target=wp,
                    property_name="image_edits",
                    new_value=new,
                    old_value=old,
                    name=name,
                )
            )
