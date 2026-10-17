import pytest

from rayforge.core.doc import Doc
from rayforge.core.variable_text import (
    CsvMerge,
    CsvRow,
    VariableText,
    parse_csv,
)
from tests.core.template_provider import TemplateProvider, place

PEOPLE = CsvMerge(
    header=("Name", "City"),
    rows=(("Ann", "Gent"), ("Bob", "Hasselt"), ("Cas", "Tongeren")),
    path="/tmp/people.csv",
)


class TestCsvRow:
    def test_by_name_and_position(self):
        row = CsvRow({"Name": "Ann", "City": "Gent"})
        assert row["Name"] == "Ann"
        assert row[0] == "Ann"
        assert row[1] == "Gent"
        assert row[-1] == "Gent"

    def test_missing_columns_are_empty(self):
        row = CsvRow({"Name": "Ann"})
        assert row["Price"] == ""
        assert row[5] == ""


class TestParseCsv:
    def test_comma_with_header(self):
        header, rows = parse_csv("Name,City\nAnn,Gent\nBob,Hasselt\n")
        assert header == ("Name", "City")
        assert rows == (("Ann", "Gent"), ("Bob", "Hasselt"))

    def test_semicolon_and_bom(self):
        header, rows = parse_csv("﻿Name;Price\nAnn;3,50\nBob;4,00\n")
        assert header == ("Name", "Price")
        assert rows[0] == ("Ann", "3,50")

    def test_tab_separated(self):
        header, rows = parse_csv("Name\tCity\nAnn\tGent\n")
        assert header == ("Name", "City")
        assert rows == (("Ann", "Gent"),)

    def test_quoted_fields(self):
        _header, rows = parse_csv('Name,Note\nAnn,"Hello, world"\n')
        assert rows == (("Ann", "Hello, world"),)

    def test_without_header_columns_are_numbered(self):
        header, rows = parse_csv("12,34\n56,78\n90,12\n")
        assert header == ("Column 1", "Column 2")
        assert rows[0] == ("12", "34")

    def test_first_row_is_the_header_unless_numeric(self):
        header, rows = parse_csv("Ann\nBob\n")
        assert header == ("Ann",)
        assert rows == (("Bob",),)

    def test_blank_lines_are_skipped(self):
        _header, rows = parse_csv("Name\nAnn\n\nBob\n")
        assert rows == (("Ann",), ("Bob",))

    def test_empty_text(self):
        assert parse_csv("") == ((), ())


class TestCsvMerge:
    def test_default_is_not_loaded(self):
        assert not CsvMerge().is_loaded

    def test_row_at_offset(self):
        merge = PEOPLE.with_current(2)
        assert merge.row_at(0)["Name"] == "Bob"
        assert merge.row_at(1)["Name"] == "Cas"
        assert merge.row_number(1) == 3

    def test_past_the_end_is_empty(self):
        row = PEOPLE.with_current(3).row_at(1)
        assert row["Name"] == ""
        assert row[1] == ""

    def test_short_rows_are_padded(self):
        merge = CsvMerge(header=("A", "B"), rows=(("x",),))
        assert merge.row_at(0)["B"] == ""

    def test_advance_stops_after_the_last_row(self):
        merge = PEOPLE.advanced(2)
        assert merge.current == 3
        assert not merge.is_exhausted
        merge = merge.advanced(5)
        assert merge.current == 4
        assert merge.is_exhausted
        assert merge.remaining == 0

    def test_back_stops_at_the_first_row(self):
        assert PEOPLE.with_current(2).advanced(-5).current == 1

    def test_remaining(self):
        assert PEOPLE.remaining == 3
        assert PEOPLE.with_current(3).remaining == 1

    def test_round_trip(self):
        merge = PEOPLE.with_current(2)
        assert CsvMerge.from_dict(merge.to_dict()) == merge

    def test_replacing_rows_keeps_the_current_row(self):
        merge = PEOPLE.with_current(2).with_rows(
            ("Name",), (("Ann",), ("Bob",), ("Cas",), ("Dirk",))
        )
        assert merge.current == 2
        assert merge.path == PEOPLE.path

    def test_shorter_rows_clamp_the_current_row(self):
        merge = PEOPLE.with_current(3).with_rows(("Name",), (("Ann",),))
        assert merge.current == 2
        assert merge.is_exhausted


class TestVariableTextCsv:
    def test_names(self):
        assert {"csv", "row", "serial"} <= VariableText().names

    def test_template_values(self):
        variables = VariableText()
        variables.csv = PEOPLE.with_current(2)
        values = variables.template_values(1)
        assert values["csv"]["City"] == "Tongeren"
        assert values["row"] == 3

    def test_setting_csv_signals(self):
        variables = VariableText()
        seen = []
        variables.changed.connect(seen.append, weak=False)
        variables.csv = PEOPLE
        assert seen == [variables]

    def test_round_trip(self):
        variables = VariableText()
        variables.csv = PEOPLE.with_current(2)
        restored = VariableText.from_dict(variables.to_dict())
        assert restored.csv == variables.csv

    def test_old_projects_have_no_csv(self):
        restored = VariableText.from_dict({"serial": {"current": 4}})
        assert not restored.csv.is_loaded


class TestTemplates:
    @pytest.fixture
    def doc(self):
        doc = Doc()
        doc.variable_text.csv = PEOPLE
        return doc

    def _rendered(self, wp):
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        return provider.rendered[-1]

    def test_column_by_name(self, doc):
        wp = place(doc, '{csv["Name"]} from {csv["City"]}')
        assert self._rendered(wp) == "Ann from Gent"

    def test_column_by_position_and_row(self, doc):
        wp = place(doc, "{row}: {csv[0]}", offset=2)
        assert self._rendered(wp) == "3: Cas"

    def test_copy_past_the_end_is_empty(self, doc):
        wp = place(doc, '[{csv["Name"]}]', offset=5)
        assert self._rendered(wp) == "[]"

    def test_changing_the_row_regenerates(self, doc):
        wp = place(doc, '{csv["Name"]}')
        doc.variable_text.csv = doc.variable_text.csv.with_current(2)
        assert self._rendered(wp) == "Bob"
