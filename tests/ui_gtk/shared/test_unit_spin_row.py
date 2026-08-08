"""UI tests for the UnitSpinRow widget family."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Adw

from rayforge.ui_gtk.shared.pref_rows import (
    AccelerationSpinRow,
    AngleSpinRow,
    LengthChoiceSpinRow,
    LengthSpinRow,
    SpeedSpinRow,
    SpinRow,
    UnitSpinRow,
)


@pytest.mark.ui
def test_angle_spin_row_defaults(ui_context_initializer):
    row = AngleSpinRow("Angle")
    adj = row.get_adjustment()
    assert adj.get_lower() == pytest.approx(-360.0)
    assert adj.get_upper() == pytest.approx(360.0)
    assert row.get_digits() == 1


@pytest.mark.ui
def test_angle_spin_row_overrides(ui_context_initializer):
    row = AngleSpinRow("Half Turn", lower=0, upper=180, digits=0)
    adj = row.get_adjustment()
    assert adj.get_lower() == pytest.approx(0.0)
    assert adj.get_upper() == pytest.approx(180.0)
    assert row.get_digits() == 0


@pytest.mark.ui
def test_uniform_entry_width_across_rows(ui_context_initializer):
    # The spin entry must have one fixed width regardless of the value
    # range or the initial value, so all rows look consistent.
    small = SpinRow("Count", lower=1, upper=8, digits=0, value=1)
    large = LengthSpinRow("Len", lower=-10000, upper=10000, value_in_base=5.0)
    tiny = LengthSpinRow("Tiny", lower=0, upper=10, value_in_base=0.05)
    widths = {
        r.get_spin_button().get_width_chars() for r in (small, large, tiny)
    }
    assert len(widths) == 1


@pytest.mark.ui
def test_plain_spinrow_get_set_value(ui_context_initializer):
    row = SpinRow("Count", lower=0, upper=100, digits=0, value=5)
    assert isinstance(row, Adw.ActionRow)
    assert row.get_value() == pytest.approx(5.0)
    assert row.get_int_value() == 5
    row.set_value(42)
    assert row.get_int_value() == 42


@pytest.mark.ui
def test_plain_spinrow_set_value_does_not_emit(ui_context_initializer):
    row = SpinRow("Count", lower=0, upper=100, value=0)
    received = []
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    row.set_value(10)
    assert received == []


@pytest.mark.ui
def test_plain_spinrow_user_edit_emits(ui_context_initializer):
    row = SpinRow("Count", lower=0, upper=100, value=0)
    received = []
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    row.get_spin_button().set_value(7)
    assert len(received) == 1


@pytest.mark.ui
def test_plain_spinrow_set_range(ui_context_initializer):
    row = SpinRow("Count", lower=0, upper=100, value=0)
    row.set_range(5, 50)
    adj = row.get_adjustment()
    assert adj.get_lower() == pytest.approx(5.0)
    assert adj.get_upper() == pytest.approx(50.0)


@pytest.mark.ui
def test_plain_spinrow_editable_delegated(ui_context_initializer):
    row = SpinRow("Count", lower=0, upper=100, value=0)
    assert row.get_editable() is True
    row.set_editable(False)
    assert row.get_editable() is False


@pytest.mark.ui
def test_is_an_action_row_with_suffixes(ui_context_initializer):
    row = LengthSpinRow("Width", lower=0, upper=1000, value_in_base=50.0)
    assert isinstance(row, Adw.ActionRow)
    # value round-trips in base units
    assert row.get_value_in_base_units() == pytest.approx(50.0)
    # the unit is shown via a tooltip, not in the subtitle
    assert row.get_subtitle() == ""
    assert row.get_spin_button() is not None


@pytest.mark.ui
def test_subtitle_is_plain_description(ui_context_initializer):
    row = LengthSpinRow("X Extent", subtitle="Full X-axis travel range")
    # plain human text; the unit is shown via a tooltip on the entry box
    assert row.get_subtitle() == "Full X-axis travel range"


@pytest.mark.ui
def test_unit_tooltip_follows_config(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=25.4)
    assert row.get_value_in_base_units() == pytest.approx(25.4)
    # mm by default
    assert row.get_spin_button().get_value() == pytest.approx(25.4)
    tooltip = row.get_spin_button().get_tooltip_text()
    assert tooltip is not None
    assert "mm" in tooltip
    # switching the display unit updates the tooltip
    ui_context_initializer.config.set_unit_preference("length", "in")
    tooltip = row.get_spin_button().get_tooltip_text()
    assert tooltip is not None
    assert "in" in tooltip
    assert row.get_value_in_base_units() == pytest.approx(25.4)


@pytest.mark.ui
def test_imperial_switch_converts_value_live(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=25.4)
    # flip the display unit to inches; the row must keep the semantic
    # value while showing it in the new unit.
    ui_context_initializer.config.set_unit_preference("length", "in")
    assert row.get_value_in_base_units() == pytest.approx(25.4)
    assert row.get_spin_button().get_value() == pytest.approx(1.0)


@pytest.mark.ui
def test_value_round_trip(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=12.5)
    assert row.get_value_in_base_units() == pytest.approx(12.5)
    row.set_value_in_base_units(42.0)
    assert row.get_value_in_base_units() == pytest.approx(42.0)


@pytest.mark.ui
def test_value_changed_signal_fires(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=0.0)
    received = []
    # weak=False: blinker holds receivers weakly by default; a bare
    # lambda with no strong reference would be collected immediately.
    # Production code connects bound methods (self.on_x_changed).
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    row.get_spin_button().set_value(5.0)
    assert received


@pytest.mark.ui
def test_set_value_programmatically_does_not_fire(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=0.0)
    received = []
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    row.set_value_in_base_units(7.0)
    # programmatic updates are not user edits
    assert received == []


@pytest.mark.ui
def test_debounce_coalesces_rapid_changes(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=0.0, debounce_ms=10)
    received = []
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    spin = row.get_spin_button()
    spin.set_value(1.0)
    spin.set_value(2.0)
    spin.set_value(3.0)
    # three rapid changes reschedule one pending timer, no emission yet
    assert received == []
    assert row._debounce_timer_id is not None
    # flushing the pending timer emits exactly once (coalesced)
    row._flush_changed()
    assert len(received) == 1


@pytest.mark.ui
def test_no_debounce_emits_immediately(ui_context_initializer):
    row = LengthSpinRow("Len", value_in_base=0.0)
    received = []
    row.value_changed.connect(lambda r: received.append(r), weak=False)
    row.get_spin_button().set_value(5.0)
    assert len(received) == 1
    assert row._debounce_timer_id is None


@pytest.mark.ui
def test_set_range(ui_context_initializer):
    ui_context_initializer.config.unit_preferences["length"] = "mm"
    row = LengthSpinRow("Len", lower=0, upper=1000)
    row.set_range(10.0, 500.0)
    adj = row.get_spin_button().get_adjustment()
    assert adj.get_lower() == pytest.approx(10.0)
    assert adj.get_upper() == pytest.approx(500.0)


@pytest.mark.ui
def test_bounds_follow_display_unit(ui_context_initializer):
    config = ui_context_initializer.config
    config.unit_preferences["length"] = "mm"
    row = LengthSpinRow("Len", lower=0, upper=25.4)
    adj = row.get_spin_button().get_adjustment()
    assert adj.get_upper() == pytest.approx(25.4)
    config.set_unit_preference("length", "in")
    assert adj.get_upper() == pytest.approx(1.0)


@pytest.mark.ui
def test_set_range_in_base_units(ui_context_initializer):
    config = ui_context_initializer.config
    config.unit_preferences["length"] = "in"
    row = LengthSpinRow("Len", lower=0, upper=1000)
    row.set_range(0.0, 25.4)
    adj = row.get_spin_button().get_adjustment()
    assert adj.get_upper() == pytest.approx(1.0)


@pytest.mark.ui
def test_added_to_preferences_group(ui_context_initializer):
    row = LengthSpinRow("Len")
    group = Adw.PreferencesGroup()
    group.add(row)


@pytest.mark.ui
def test_quantity_subclasses(ui_context_initializer):
    speed = SpeedSpinRow("Speed", value_in_base=1000.0)
    accel = AccelerationSpinRow("Accel", value_in_base=500.0)
    assert isinstance(speed, UnitSpinRow)
    assert isinstance(accel, UnitSpinRow)
    assert speed.quantity == "speed"
    assert accel.quantity == "acceleration"


@pytest.mark.ui
def test_edit_text_survives_config_round_trip(ui_context_initializer):
    """A config change triggered by an edit must not reformat the text.

    Editing a row fires value_changed, which a consumer (e.g. a machine
    preference page) may answer by mutating the machine. Machine changes
    propagate through config.changed, which re-enters the row. The row
    must not rewrite its text then, or the cursor yanks mid-edit.
    """
    config = ui_context_initializer.config
    row = LengthSpinRow(
        "Len",
        lower=0.01,
        upper=10.0,
        value_in_base=0.1,
    )
    row.value_changed.connect(
        lambda r: config.changed.send(config), weak=False
    )
    spin = row.get_spin_button()
    spin.select_region(0, -1)
    spin.delete_selection()
    assert spin.get_text() == ""
    for text in ("0", "0.", "0.1"):
        spin.set_text(text)
        assert spin.get_text() == text


@pytest.mark.ui
def test_display_unit_switch_still_reformats(ui_context_initializer):
    """A genuine display-unit switch still rewrites the text."""
    config = ui_context_initializer.config
    row = LengthSpinRow("Len", lower=0, upper=1000, value_in_base=25.4)
    config.set_unit_preference("length", "in")
    assert row.get_value_in_base_units() == pytest.approx(25.4)
    assert row.get_spin_button().get_text() == "1.000"


@pytest.mark.ui
def test_length_choice_defaults_to_preferred_unit(ui_context_initializer):
    config = ui_context_initializer.config
    config.unit_preferences["length"] = "mm"
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    unit = row._units[row._unit_dropdown.get_selected()]
    assert unit.name == "mm"
    assert row.get_spin_button().get_value() == pytest.approx(25.4)
    assert row.get_value_in_base_units() == pytest.approx(25.4)


@pytest.mark.ui
def test_length_choice_dropdown_follows_config_pref(ui_context_initializer):
    config = ui_context_initializer.config
    config.unit_preferences["length"] = "in"
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    unit = row._units[row._unit_dropdown.get_selected()]
    assert unit.name == "in"
    assert row.get_spin_button().get_value() == pytest.approx(1.0)
    assert row.get_value_in_base_units() == pytest.approx(25.4)


@pytest.mark.ui
def test_length_choice_unit_switch_preserves_base_value(
    ui_context_initializer,
):
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    row._unit_dropdown.set_selected(row._unit_index("in"))
    assert row._unit is not None
    assert row._unit.name == "in"
    assert row.get_spin_button().get_value() == pytest.approx(1.0)
    assert row.get_value_in_base_units() == pytest.approx(25.4)


@pytest.mark.ui
def test_length_choice_unit_switch_keeps_global_pref(ui_context_initializer):
    config = ui_context_initializer.config
    config.unit_preferences["length"] = "mm"
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    row._unit_dropdown.set_selected(row._unit_index("in"))
    assert config.unit_preferences["length"] == "mm"
    assert row._unit is not None
    assert row._unit.name == "in"


@pytest.mark.ui
def test_length_choice_override_survives_pref_change(ui_context_initializer):
    config = ui_context_initializer.config
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    row._unit_dropdown.set_selected(row._unit_index("in"))
    assert row._unit is not None
    assert row._unit.name == "in"
    config.set_unit_preference("length", "cm")
    assert row._unit is not None
    assert row._unit.name == "in"
    assert row.get_spin_button().get_value() == pytest.approx(1.0)
    assert row.get_value_in_base_units() == pytest.approx(25.4)


@pytest.mark.ui
def test_length_choice_dropdown_is_attached_suffix(ui_context_initializer):
    row = LengthChoiceSpinRow("Len", value_in_base=25.4)
    spin = row.get_spin_button()
    dd = row._unit_dropdown
    # both widgets live inside the row (suffixes are wrapped in a box)
    assert spin.is_ancestor(row)
    assert dd.is_ancestor(row)
    model = dd.get_model()
    assert model is not None
    assert model.get_n_items() == len(row._units)
    # the dropdown sits immediately to the right of the spin button
    assert spin.get_next_sibling() is dd
