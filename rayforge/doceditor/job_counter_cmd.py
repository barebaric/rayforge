"""
JobCounterCmd: undoable edits of the document's job counter and of
the per-workpiece variable offsets.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..core.item import DocItem
from ..core.job_counter import CounterSettings
from ..core.undo import ChangePropertyCommand
from ..core.workpiece import WorkPiece, workpieces_in

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class JobCounterCmd:
    """Handles changes to the document's job counter."""

    def __init__(self, editor: DocEditor):
        self._editor = editor
        editor.file.gcode_exported.connect(self._on_gcode_exported)

    @property
    def counter(self) -> CounterSettings:
        return self._editor.doc.job_counter.counter

    def set_counter(
        self, settings: CounterSettings, name: str | None = None
    ) -> None:
        """Replaces the job counter settings in one undoable step."""
        job_counter = self._editor.doc.job_counter
        if settings == job_counter.counter:
            return
        cmd = ChangePropertyCommand(
            target=job_counter,
            property_name="counter",
            new_value=settings,
            name=name or _("Change counter value"),
        )
        self._editor.history_manager.execute(cmd)

    def next_counter(self) -> None:
        """Moves the counter forward by one job."""
        copies = max(self._editor.doc.job_counter_copies(), 1)
        self.set_counter(
            self.counter.advanced(copies), _("Next counter value")
        )

    def previous_counter(self) -> None:
        """Moves the counter back by one job."""
        copies = max(self._editor.doc.job_counter_copies(), 1)
        self.set_counter(
            self.counter.advanced(-copies), _("Previous counter value")
        )

    def reset_counter(self) -> None:
        """Puts the counter back at its start value."""
        self.set_counter(self.counter.reset(), _("Reset counter value"))

    def preview_counter_values(self) -> list[int]:
        """The counter values the next job uses, one per copy."""
        copies = max(self._editor.doc.job_counter_copies(), 1)
        return self.counter.values_for(copies)

    def advance_after_job(self, exported: bool = False) -> bool:
        """
        Moves the counter on after a finished job (or a G-code export,
        if *exported*), by one value per copy in the job.

        Returns:
            True if the counter was advanced.
        """
        counter = self.counter
        enabled = (
            counter.advance_on_export if exported else counter.auto_advance
        )
        copies = self._editor.doc.job_counter_copies()
        if not enabled or copies == 0:
            return False
        self.set_counter(
            counter.advanced(copies), _("Advance counter value after job")
        )
        logger.info(
            f"Job counter advanced by {copies} to "
            f"{self._editor.doc.job_counter.counter.current}"
        )
        return True

    def on_job_completed(self, sender) -> None:
        """Advances the counter after a job sent to the machine."""
        self.advance_after_job()

    def _on_gcode_exported(self, sender) -> None:
        self.advance_after_job(exported=True)

    def number_copies(self, items: Sequence[DocItem]) -> None:
        """
        Gives *items* the offsets 0, 1, 2, ... in reading order: rows
        from top to bottom, each row from left to right. All workpieces
        inside a group share the group's offset.
        """
        changes = []
        for index, item in enumerate(_reading_order(items)):
            changes.extend((wp, index) for wp in workpieces_in(item))
        self._apply_offsets(changes, _("Number copies"))

    def _apply_offsets(
        self, changes: list[tuple[WorkPiece, int]], name: str
    ) -> None:
        history = self._editor.history_manager
        with history.transaction(name) as t:
            for wp, offset in changes:
                if wp.variable_offset == offset:
                    continue
                t.execute(
                    ChangePropertyCommand(
                        target=wp,
                        property_name="variable_offset",
                        new_value=offset,
                        name=name,
                    )
                )


def _reading_order(items: Sequence[DocItem]) -> list[DocItem]:
    """Sorts items into rows (top first), then left to right."""
    boxes = []
    for item in items:
        x, y, width, height = item.bbox
        boxes.append((item, x + width / 2, y + height / 2, height))
    boxes.sort(key=lambda b: -b[2])

    rows: list[list[tuple[DocItem, float, float, float]]] = []
    for box in boxes:
        if rows:
            first = rows[-1][0]
            tolerance = max(min(first[3], box[3]) / 2, 1e-6)
            if abs(first[2] - box[2]) <= tolerance:
                rows[-1].append(box)
                continue
        rows.append([box])

    ordered: list[DocItem] = []
    for row in rows:
        ordered.extend(b[0] for b in sorted(row, key=lambda b: b[1]))
    return ordered
