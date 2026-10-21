"""
VariableTextCmd: undoable edits of the document's template variables
(serial counter, merged CSV rows) and of the per-workpiece variable
offsets.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..core.item import DocItem
from ..core.undo import ChangePropertyCommand
from ..core.variable_text import (
    CSV_NAME,
    ROW_NAME,
    SERIAL_NAME,
    CsvMerge,
    SerialSettings,
    read_csv_file,
)
from ..core.workpiece import WorkPiece
from .layout.base import LayoutStrategy

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class VariableTextCmd:
    """Handles changes to the document's variable text settings."""

    def __init__(self, editor: DocEditor):
        self._editor = editor

    @property
    def serial(self) -> SerialSettings:
        return self._editor.doc.variable_text.serial

    def set_serial(
        self, settings: SerialSettings, name: str | None = None
    ) -> None:
        """Replaces the serial counter settings in one undoable step."""
        variable_text = self._editor.doc.variable_text
        if settings == variable_text.serial:
            return
        cmd = ChangePropertyCommand(
            target=variable_text,
            property_name="serial",
            new_value=settings,
            name=name or _("Change serial number"),
        )
        self._editor.history_manager.execute(cmd)

    def next_serial(self) -> None:
        """Moves the counter forward by one job."""
        copies = max(self._editor.doc.variable_text_copies(), 1)
        self.set_serial(self.serial.advanced(copies), _("Next serial number"))

    def previous_serial(self) -> None:
        """Moves the counter back by one job."""
        copies = max(self._editor.doc.variable_text_copies(), 1)
        self.set_serial(
            self.serial.advanced(-copies), _("Previous serial number")
        )

    def reset_serial(self) -> None:
        """Puts the counter back at its start value."""
        self.set_serial(self.serial.reset(), _("Reset serial number"))

    def preview_serial_values(self) -> list[int]:
        """The serial numbers the next job uses, one per copy."""
        copies = max(self._editor.doc.variable_text_copies(), 1)
        return self.serial.values_for(copies)

    def advance_after_job(self, exported: bool = False) -> bool:
        """
        Moves the variables on after a finished job (or a G-code export,
        if *exported*), by one value per copy in the job, in one undo
        step. The serial counter and the CSV row only move when a
        workpiece uses them; the CSV row is not moved by an export.

        Returns:
            True if anything was advanced.
        """
        variables = self._editor.doc.variable_text
        serial, merge = variables.serial, variables.csv
        new_serial = self._advanced_serial(serial, exported)
        new_merge = None if exported else self._advanced_csv(merge)
        if new_serial is None and new_merge is None:
            return False
        history = self._editor.history_manager
        with history.transaction(_("Advance variable text after job")) as t:
            if new_serial is not None:
                t.execute(self._change("serial", new_serial))
            if new_merge is not None:
                t.execute(self._change("csv", new_merge))
        if new_merge is not None and new_merge.is_exhausted:
            self._notify(
                _("All {count} CSV rows have been used.").format(
                    count=new_merge.row_count
                )
            )
        logger.info(
            f"Variables advanced after job: serial {serial.current} -> "
            f"{variables.serial.current}, CSV row {merge.current} -> "
            f"{variables.csv.current}"
        )
        return True

    def _advanced_serial(
        self, serial: SerialSettings, exported: bool
    ) -> SerialSettings | None:
        enabled = serial.advance_on_export if exported else serial.auto_advance
        copies = self._editor.doc.variable_text_copies({SERIAL_NAME})
        if not enabled or copies == 0:
            return None
        return serial.advanced(copies)

    def _advanced_csv(self, merge: CsvMerge) -> CsvMerge | None:
        copies = self._csv_copies()
        if not merge.is_loaded or not merge.auto_advance or copies == 0:
            return None
        advanced = merge.advanced(copies)
        return None if advanced == merge else advanced

    def _change(self, prop: str, value: object) -> ChangePropertyCommand:
        return ChangePropertyCommand(
            target=self._editor.doc.variable_text,
            property_name=prop,
            new_value=value,
        )

    def _notify(self, message: str) -> None:
        self._editor.notification_requested.send(self, message=message)

    @property
    def csv(self) -> CsvMerge:
        return self._editor.doc.variable_text.csv

    def _csv_copies(self) -> int:
        return self._editor.doc.variable_text_copies({CSV_NAME, ROW_NAME})

    def set_csv(self, merge: CsvMerge, name: str | None = None) -> None:
        """Replaces the merged CSV in one undoable step."""
        variable_text = self._editor.doc.variable_text
        if merge == variable_text.csv:
            return
        cmd = ChangePropertyCommand(
            target=variable_text,
            property_name="csv",
            new_value=merge,
            name=name or _("Change CSV merge"),
        )
        self._editor.history_manager.execute(cmd)

    def _read(self, path: str) -> CsvMerge | None:
        try:
            return read_csv_file(path)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read CSV file {path}: {e}")
            self._notify(
                _("Could not read the CSV file {path}: {error}").format(
                    path=path, error=e
                )
            )
            return None

    def load_csv(self, path: str) -> bool:
        """Links a CSV file and stores its rows in the project."""
        loaded = self._read(path)
        if loaded is None:
            return False
        merge = self.csv
        self.set_csv(
            CsvMerge(
                header=loaded.header,
                rows=loaded.rows,
                path=path,
                auto_advance=merge.auto_advance,
            ),
            _("Load CSV file"),
        )
        return True

    def reload_csv(self) -> bool:
        """Reads the linked file again, keeping the current row."""
        merge = self.csv
        if not merge.path:
            return False
        loaded = self._read(merge.path)
        if loaded is None:
            return False
        self.set_csv(
            merge.with_rows(loaded.header, loaded.rows), _("Reload CSV file")
        )
        return True

    def first_row(self) -> None:
        """Goes back to the first row."""
        self.set_csv(self.csv.with_current(1), _("First CSV row"))

    def next_row(self) -> None:
        """Moves the row forward by one job."""
        copies = max(self._csv_copies(), 1)
        self.set_csv(self.csv.advanced(copies), _("Next CSV row"))

    def previous_row(self) -> None:
        """Moves the row back by one job."""
        copies = max(self._csv_copies(), 1)
        self.set_csv(self.csv.advanced(-copies), _("Previous CSV row"))

    def preview_rows(self) -> list[int]:
        """The 1-based rows the next job uses, one per copy."""
        copies = max(self._csv_copies(), 1)
        return [self.csv.row_number(i) for i in range(copies)]

    def copies_needed(self) -> int:
        """How many copies cover the rows that are left."""
        return self.csv.remaining

    def set_variable_offset(
        self, workpieces: Sequence[WorkPiece], offset: int
    ) -> None:
        """Sets the same variable offset on all *workpieces*."""
        self._apply_offsets(
            [(wp, offset) for wp in workpieces], _("Set variable offset")
        )

    def number_copies(self, items: Sequence[DocItem]) -> None:
        """
        Gives *items* the offsets 0, 1, 2, ... in reading order: rows
        from top to bottom, each row from left to right. All workpieces
        inside a group share the group's offset.
        """
        changes = []
        for index, item in enumerate(_reading_order(items)):
            changes.extend((wp, index) for wp in _workpieces_of(item))
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


def _workpieces_of(item: DocItem) -> list[WorkPiece]:
    if isinstance(item, WorkPiece):
        return [item]
    return item.get_descendants(of_type=WorkPiece)


def _reading_order(items: Sequence[DocItem]) -> list[DocItem]:
    """Sorts items into rows (top first), then left to right."""
    boxes = []
    for item in items:
        bbox = LayoutStrategy._get_item_world_bbox(item)
        if bbox is None:
            x, y = item.pos
            bbox = (x, y, x, y)
        x_min, y_min, x_max, y_max = bbox
        boxes.append(
            (item, (x_min + x_max) / 2, (y_min + y_max) / 2, y_max - y_min)
        )
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
