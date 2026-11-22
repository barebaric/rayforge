"""Tests for the bidirectional scan offset table in Machine Settings."""

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.advanced_preferences_page import (
    AdvancedPreferencesPage,
)
from rayforge.ui_gtk.machine.bidir_offset_table import BidirOffsetTableGroup

pytestmark = pytest.mark.ui


def _group(machine) -> BidirOffsetTableGroup:
    page = AdvancedPreferencesPage(machine)
    assert isinstance(page.bidir_offset_group, BidirOffsetTableGroup)
    return page.bidir_offset_group


def test_empty_table_shows_placeholder(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    group = _group(machine)

    assert group.entry_rows == []
    assert group.empty_row.get_visible() is True


def test_rows_mirror_the_machine_table(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.set_bidir_offset_table([(1000, 0.1), (3000, 0.3)])
    group = _group(machine)

    assert len(group.entry_rows) == 2
    assert group.empty_row.get_visible() is False
    first = group.entry_rows[0]
    assert first.speed_row.get_value_in_base_units() == pytest.approx(1000)
    assert first.offset_row.get_value_in_base_units() == pytest.approx(0.1)


def test_add_appends_a_row_above_the_fastest_speed(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.set_bidir_offset_table([(1000, 0.1)])
    group = _group(machine)

    group.add_button.emit("clicked")

    assert machine.bidir_offset_table == [(1000.0, 0.1), (2000.0, 0.1)]
    assert len(group.entry_rows) == 2


def test_add_on_empty_table_uses_max_cut_speed(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.max_cut_speed = 3000
    group = _group(machine)

    group.add_button.emit("clicked")

    assert machine.bidir_offset_table == [(3000.0, 0.0)]


def test_editing_a_row_updates_the_machine(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.set_bidir_offset_table([(1000, 0.1), (3000, 0.3)])
    group = _group(machine)

    group.entry_rows[1].offset_row.set_value_in_base_units(0.25)
    group.entry_rows[1].offset_row.value_changed.send(
        group.entry_rows[1].offset_row
    )

    assert machine.bidir_offset_table == [(1000.0, 0.1), (3000.0, 0.25)]


def test_remove_drops_the_row(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.set_bidir_offset_table([(1000, 0.1), (3000, 0.3)])
    group = _group(machine)

    group.entry_rows[0].remove_button.emit("clicked")

    assert machine.bidir_offset_table == [(3000.0, 0.3)]
    assert len(group.entry_rows) == 1
