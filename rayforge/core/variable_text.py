"""
Document-level variables for text templates.

A document carries a :class:`VariableText` instance whose values are
injected into the ``{...}`` templates of every geometry provider (for
example the text boxes of a sketch). Each workpiece adds its own
*variable offset*, so copies of the same design can show consecutive
serial numbers or consecutive rows of a merged CSV file.
"""

from __future__ import annotations

import ast
import csv
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from gettext import gettext as _
from typing import Any

from blinker import Signal

SERIAL_NAME = "serial"
CSV_NAME = "csv"
ROW_NAME = "row"

_CSV_DELIMITERS = ",;\t|"

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class SerialSettings:
    """
    An immutable serial counter.

    Attributes:
        current: The value the next job starts with.
        start: The value :meth:`reset` returns to, and where the
            counter wraps to after ``end``.
        step: How much the value grows per copy and per job.
        end: Optional last value. Values past it wrap to ``start``.
        auto_advance: Whether a finished job moves ``current`` on.
        advance_on_export: Whether exporting G-code moves ``current``
            on as well, for jobs that run from a memory card.
    """

    current: int = 1
    start: int = 1
    step: int = 1
    end: int | None = None
    auto_advance: bool = True
    advance_on_export: bool = False

    def _range_length(self) -> int | None:
        """Number of values in ``start..end``, or None if unbounded."""
        if self.end is None or self.step <= 0 or self.end < self.start:
            return None
        return (self.end - self.start) // self.step + 1

    def value_at(self, offset: int) -> int:
        """The value shown by a copy that is *offset* copies further."""
        value = self.current + offset * self.step
        length = self._range_length()
        if length is None:
            return value
        index = (value - self.start) // self.step
        return self.start + (index % length) * self.step

    def values_for(self, copies: int) -> list[int]:
        """The values of *copies* consecutive copies."""
        return [self.value_at(i) for i in range(max(copies, 0))]

    def advanced(self, copies: int = 1) -> SerialSettings:
        """A counter moved on (or back, if negative) by *copies*."""
        return replace(self, current=self.value_at(copies))

    def reset(self) -> SerialSettings:
        """A counter whose current value is back at ``start``."""
        return replace(self, current=self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "start": self.start,
            "step": self.step,
            "end": self.end,
            "auto_advance": self.auto_advance,
            "advance_on_export": self.advance_on_export,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerialSettings:
        default = cls()
        start = int(data.get("start", default.start))
        end = data.get("end")
        if end is not None:
            end = int(end)
            if end < start:
                end = None
        return cls(
            current=int(data.get("current", default.current)),
            start=start,
            step=int(data.get("step", default.step)),
            end=end,
            auto_advance=bool(data.get("auto_advance", default.auto_advance)),
            advance_on_export=bool(
                data.get("advance_on_export", default.advance_on_export)
            ),
        )


class CsvRow(dict):
    """
    One CSV row as templates see it: ``csv["Name"]`` by column name or
    ``csv[0]`` by position. A missing column reads as the empty string.
    """

    def __missing__(self, key: Any) -> str:
        if isinstance(key, int) and not isinstance(key, bool):
            values = list(self.values())
            if -len(values) <= key < len(values):
                return values[key]
        return ""


def _sniff_dialect(text: str) -> type[csv.Dialect] | csv.Dialect:
    try:
        return csv.Sniffer().sniff(text[:65536], delimiters=_CSV_DELIMITERS)
    except csv.Error:
        return csv.excel


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", "."))
    except ValueError:
        return False
    return True


def parse_csv(
    text: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """
    Splits CSV text into a header and rows, detecting the delimiter and
    whether the first row is a header. Without a header the columns are
    named "Column 1", "Column 2", ... Blank lines are skipped.
    """
    text = text.removeprefix("\ufeff")
    if not text.strip():
        return (), ()
    dialect = _sniff_dialect(text)
    records = [
        tuple(cell.strip() for cell in record)
        for record in csv.reader(io.StringIO(text), dialect)
        if any(cell.strip() for cell in record)
    ]
    if not records:
        return (), ()
    width = max(len(record) for record in records)
    if not all(_is_number(cell) for cell in records[0]):
        header = records[0] + ("",) * (width - len(records[0]))
        header = tuple(
            name or _("Column {number}").format(number=i + 1)
            for i, name in enumerate(header)
        )
        return header, tuple(records[1:])
    header = tuple(
        _("Column {number}").format(number=i + 1) for i in range(width)
    )
    return header, tuple(records)


def read_csv_file(path: str) -> CsvMerge:
    """
    Reads a UTF-8 CSV file (with or without BOM) into a merge source.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not valid UTF-8.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        header, rows = parse_csv(f.read())
    return CsvMerge(header=header, rows=rows, path=path)


@dataclass(frozen=True)
class CsvMerge:
    """
    The rows of a CSV file merged into text templates.

    Attributes:
        header: The column names.
        rows: The rows, stored in the project so it stays
            self-contained.
        path: The file the rows came from, for reloading.
        current: The 1-based row the next job starts with. One past the
            last row means every row has been used.
        auto_advance: Whether a finished job moves ``current`` on.
    """

    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    path: str | None = None
    current: int = 1
    auto_advance: bool = True

    @property
    def is_loaded(self) -> bool:
        return bool(self.rows)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def remaining(self) -> int:
        """Rows from ``current`` to the end."""
        return max(self.row_count - self.current + 1, 0)

    @property
    def is_exhausted(self) -> bool:
        """Whether every row has been used."""
        return self.is_loaded and self.remaining == 0

    def row_number(self, offset: int) -> int:
        """The 1-based row a copy that is *offset* copies further shows."""
        return self.current + offset

    def row_at(self, offset: int) -> CsvRow:
        """The row a copy shows; past the end every column is empty."""
        index = self.row_number(offset) - 1
        if 0 <= index < self.row_count:
            cells = self.rows[index]
        else:
            cells = ()
        return CsvRow(
            (name, cells[i] if i < len(cells) else "")
            for i, name in enumerate(self.header)
        )

    def with_current(self, row: int) -> CsvMerge:
        """The same rows with ``current`` clamped to the valid range."""
        last = self.row_count + 1 if self.is_loaded else 1
        return replace(self, current=min(max(row, 1), last))

    def advanced(self, copies: int = 1) -> CsvMerge:
        """Moved on (or back, if negative); stops after the last row."""
        return self.with_current(self.current + copies)

    def with_rows(
        self, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]
    ) -> CsvMerge:
        """New file contents; ``current`` is kept when still valid."""
        return replace(self, header=header, rows=rows).with_current(
            self.current
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": list(self.header),
            "rows": [list(row) for row in self.rows],
            "path": self.path,
            "current": self.current,
            "auto_advance": self.auto_advance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CsvMerge:
        merge = cls(
            header=tuple(str(name) for name in data.get("header", ())),
            rows=tuple(
                tuple(str(cell) for cell in row)
                for row in data.get("rows", ())
            ),
            path=data.get("path"),
            auto_advance=bool(data.get("auto_advance", True)),
        )
        return merge.with_current(int(data.get("current", 1)))


class VariableText:
    """
    The variables a document offers to text templates.

    Emits :attr:`changed` whenever a value changes, so workpieces whose
    templates reference a variable can regenerate.
    """

    def __init__(self):
        self.changed = Signal()
        self._serial = SerialSettings()
        self._csv = CsvMerge()

    @property
    def serial(self) -> SerialSettings:
        return self._serial

    @serial.setter
    def serial(self, value: SerialSettings) -> None:
        if value == self._serial:
            return
        self._serial = value
        self.changed.send(self)

    @property
    def csv(self) -> CsvMerge:
        return self._csv

    @csv.setter
    def csv(self, value: CsvMerge) -> None:
        if value == self._csv:
            return
        self._csv = value
        self.changed.send(self)

    @property
    def names(self) -> set[str]:
        """The template names this object provides."""
        return {SERIAL_NAME, CSV_NAME, ROW_NAME}

    def template_values(self, offset: int = 0) -> dict[str, Any]:
        """The template values for a copy with the given offset."""
        return {
            SERIAL_NAME: self._serial.value_at(offset),
            CSV_NAME: self._csv.row_at(offset),
            ROW_NAME: self._csv.row_number(offset),
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"serial": self._serial.to_dict()}
        if self._csv != CsvMerge():
            data["csv"] = self._csv.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VariableText:
        instance = cls()
        if data:
            serial = data.get("serial")
            if isinstance(serial, dict):
                instance._serial = SerialSettings.from_dict(serial)
            merge = data.get("csv")
            if isinstance(merge, dict):
                instance._csv = CsvMerge.from_dict(merge)
        return instance


def _names_in_expression(expression: str) -> set[str]:
    """Names an expression reads, ignoring string literals."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return set(_IDENTIFIER_RE.findall(expression))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def referenced_template_names(templates: Iterable[str]) -> set[str]:
    """
    Collects the names used inside the ``{...}`` placeholders of
    *templates*. A format spec after the first colon is ignored.
    """
    names: set[str] = set()
    for template in templates:
        for match in _PLACEHOLDER_RE.finditer(template or ""):
            expression, _sep, _spec = match.group(1).partition(":")
            names |= _names_in_expression(expression)
    return names
