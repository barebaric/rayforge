"""
Tests for the MoveToPopover coordinate entry and move shortcut logic.

GTK initialization is bypassed; the rows and buttons are replaced with
mocks so only the value handling and gating logic is exercised.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.models.machine_panel import PanelOrientation


def make_popover(machine, machine_cmd) -> Any:
    from rayforge.ui_gtk.machine.move_to_popover import MoveToPopover

    popover: Any = MoveToPopover.__new__(MoveToPopover)
    popover.machine = machine
    popover.machine_cmd = machine_cmd
    popover._get_bounds_callback = None
    popover._get_speed_callback = None

    popover.x_row = MagicMock()
    popover.x_row.get_value_in_base_units.return_value = 10.0
    popover.y_row = MagicMock()
    popover.y_row.get_value_in_base_units.return_value = 20.0
    popover.z_row = MagicMock()
    popover.z_row.get_value_in_base_units.return_value = 5.0
    popover.move_btn = MagicMock()
    popover.ll_btn = MagicMock()
    popover.center_btn = MagicMock()
    popover.ur_btn = MagicMock()
    popover.origin_btn = MagicMock()
    popover.alignment_row = MagicMock()
    popover._alignment_handler_id = 0
    popover.travel_warning_label = MagicMock()
    return popover


@pytest.fixture
def machine():
    m = MagicMock()
    m.has_z_axis = True
    m.is_connected.return_value = True
    m.axis_extents = (400.0, 300.0)
    m.reverse_x_axis = False
    m.reverse_y_axis = False
    m.pointer_alignment_enabled = False
    m.get_pointer_offset.return_value = (0.0, 0.0)
    m.get_command_wcs_offset.return_value = (0.0, 0.0, 0.0)
    m.get_active_wcs_offset.return_value = (0.0, 0.0, 0.0)
    m.has_pointer_offset.return_value = False
    z_cfg = MagicMock()
    z_cfg.extents = (0.0, 100.0)
    m.axes.get.return_value = z_cfg
    return m


@pytest.fixture
def machine_cmd():
    return MagicMock()


@pytest.fixture
def move_popover(sync_machine):
    """A popover wired to a real 400x300 machine with selection bounds."""
    from rayforge.ui_gtk.machine.move_to_popover import MoveToPopover

    sync_machine.set_axis_extents(400, 300)

    popover: Any = MoveToPopover.__new__(MoveToPopover)
    popover.machine = sync_machine
    popover.machine_cmd = MagicMock()
    popover._get_bounds_callback = MagicMock(
        return_value=(100.0, 50.0, 120.0, 60.0)
    )
    popover._get_speed_callback = None
    popover.alignment_row = MagicMock()
    popover._alignment_handler_id = 0
    popover.travel_warning_label = MagicMock()
    return popover


def enable_pointer_alignment(machine, x=10.0, y=20.0):
    """Configures an enabled pointer offset and turns alignment on."""
    head = machine.get_default_laser_head()
    assert head is not None
    head.set_pointer_offset(x, y)
    head.set_pointer_offset_enabled(True)
    machine.set_pointer_alignment(True)


@pytest.mark.ui
def test_move_clicked_passes_xyz(machine, machine_cmd):
    popover = make_popover(machine, machine_cmd)

    popover._issue_move()

    machine_cmd.move_to.assert_called_once_with(
        machine, 10.0, 20.0, 5.0, speed=None
    )


@pytest.mark.ui
def test_move_clicked_without_z_axis_passes_none(machine, machine_cmd):
    machine.has_z_axis = False
    popover = make_popover(machine, machine_cmd)

    popover._issue_move()

    machine_cmd.move_to.assert_called_once_with(
        machine, 10.0, 20.0, None, speed=None
    )


@pytest.mark.ui
def test_move_clicked_passes_speed_getter_value(machine, machine_cmd):
    popover = make_popover(machine, machine_cmd)
    popover.set_speed_getter(lambda: 6000)

    popover._issue_move()

    machine_cmd.move_to.assert_called_once_with(
        machine, 10.0, 20.0, 5.0, speed=6000
    )


@pytest.mark.ui
def test_move_clicked_without_machine_is_noop(machine_cmd):
    popover = make_popover(None, machine_cmd)

    popover._issue_move()

    machine_cmd.move_to.assert_not_called()


@pytest.mark.ui
def test_move_clicked_without_machine_cmd_is_noop(machine):
    popover = make_popover(machine, None)

    popover._issue_move()


@pytest.mark.ui
def test_sensitivity_connected(machine, machine_cmd):
    popover = make_popover(machine, machine_cmd)

    popover.update_sensitivity()

    popover.move_btn.set_sensitive.assert_called_with(True)
    popover.origin_btn.set_sensitive.assert_called_with(True)
    popover.ll_btn.set_sensitive.assert_called_with(False)
    popover.center_btn.set_sensitive.assert_called_with(False)
    popover.ur_btn.set_sensitive.assert_called_with(False)


@pytest.mark.ui
def test_sensitivity_offline(machine, machine_cmd):
    machine.is_connected.return_value = False
    popover = make_popover(machine, machine_cmd)

    popover.update_sensitivity()

    popover.move_btn.set_sensitive.assert_called_once_with(False)
    popover.origin_btn.set_sensitive.assert_called_once_with(False)


@pytest.mark.ui
def test_sensitivity_dummy_driver(machine, machine_cmd):
    machine.is_connected.return_value = False
    machine.driver = MagicMock(spec=NoDeviceDriver)
    popover = make_popover(machine, machine_cmd)

    popover.update_sensitivity()

    popover.move_btn.set_sensitive.assert_called_once_with(True)


@pytest.mark.ui
def test_sensitivity_with_bounds(machine, machine_cmd):
    popover = make_popover(machine, machine_cmd)
    popover.set_get_bounds_callback(
        MagicMock(return_value=(0.0, 0.0, 10.0, 10.0))
    )

    popover.update_sensitivity()

    popover.ll_btn.set_sensitive.assert_called_once_with(True)
    popover.center_btn.set_sensitive.assert_called_once_with(True)
    popover.ur_btn.set_sensitive.assert_called_once_with(True)


@pytest.mark.ui
def test_set_machine_tracks_connection_changes(machine, machine_cmd):
    from blinker import Signal

    machine.state_changed = Signal()
    machine.connection_status_changed = Signal()
    machine.changed = Signal()
    popover = make_popover(machine, machine_cmd)
    popover.machine = None

    popover.set_machine(machine, machine_cmd)
    popover.move_btn.set_sensitive.assert_called_with(True)

    machine.is_connected.return_value = False
    machine.connection_status_changed.send(machine, status=None)

    popover.move_btn.set_sensitive.assert_called_with(False)


@pytest.mark.ui
def test_bounds_use_machine_extents(machine, machine_cmd):
    popover = make_popover(machine, machine_cmd)

    popover._update_bounds_and_axes()

    popover.x_row.set_range.assert_called_once_with(0.0, 400.0)
    popover.y_row.set_range.assert_called_once_with(0.0, 300.0)
    popover.z_row.set_range.assert_called_once_with(0.0, 100.0)
    popover.z_row.set_visible.assert_called_once_with(True)


@pytest.mark.ui
def test_bounds_use_negative_z_extents(machine, machine_cmd):
    machine.axes.get.return_value.extents = (-50.0, 50.0)
    popover = make_popover(machine, machine_cmd)

    popover._update_bounds_and_axes()

    popover.z_row.set_range.assert_called_once_with(-50.0, 50.0)


@pytest.mark.ui
def test_prefill_subtracts_active_wcs_offset(machine, machine_cmd):
    machine.device_state.machine_pos = (110.0, 120.0, 8.0)
    machine.machine_space_wcs = "G53"
    machine.active_wcs = "G54"
    machine.get_wcs_offset.return_value = (10.0, 20.0, 3.0)
    popover = make_popover(machine, machine_cmd)

    popover._prefill_from_machine()

    popover.x_row.set_value_in_base_units.assert_called_once_with(100.0)
    popover.y_row.set_value_in_base_units.assert_called_once_with(100.0)
    popover.z_row.set_value_in_base_units.assert_called_once_with(5.0)


@pytest.mark.ui
def test_prefill_raw_position_in_machine_space(machine, machine_cmd):
    machine.device_state.machine_pos = (110.0, 120.0, 8.0)
    machine.machine_space_wcs = "G53"
    machine.active_wcs = "G53"
    popover = make_popover(machine, machine_cmd)

    popover._prefill_from_machine()

    popover.x_row.set_value_in_base_units.assert_called_once_with(110.0)
    popover.y_row.set_value_in_base_units.assert_called_once_with(120.0)
    popover.z_row.set_value_in_base_units.assert_called_once_with(8.0)


@pytest.mark.parametrize(
    "orientation, position, expected",
    [
        (PanelOrientation.NATIVE, "ll", (100.0, 50.0)),
        (PanelOrientation.NATIVE, "center", (110.0, 55.0)),
        (PanelOrientation.NATIVE, "ur", (120.0, 60.0)),
        (PanelOrientation.ROTATED_RIGHT, "ll", (120.0, 50.0)),
        (PanelOrientation.ROTATED_RIGHT, "center", (110.0, 55.0)),
        (PanelOrientation.ROTATED_RIGHT, "ur", (100.0, 60.0)),
        (PanelOrientation.ROTATED_LEFT, "ll", (100.0, 60.0)),
        (PanelOrientation.ROTATED_LEFT, "center", (110.0, 55.0)),
        (PanelOrientation.ROTATED_LEFT, "ur", (120.0, 50.0)),
    ],
)
@pytest.mark.ui
def test_move_to_position_honors_panel(
    move_popover, orientation, position, expected
):
    """The shortcuts target the presented corners of the selection."""
    move_popover.machine.panel.set_orientation(orientation)

    move_popover._on_move_to_position(None, position)

    move_popover.machine_cmd.move_to.assert_called_once_with(
        move_popover.machine, *expected, speed=None
    )


@pytest.mark.ui
def test_move_to_position_no_bounds_returns(move_popover):
    move_popover._get_bounds_callback = MagicMock(return_value=None)

    move_popover._on_move_to_position(None, "ll")

    move_popover.machine_cmd.move_to.assert_not_called()


@pytest.mark.ui
def test_move_to_wcs_zero(move_popover):
    move_popover._on_move_to_wcs_zero(None)

    move_popover.machine_cmd.move_to.assert_called_once_with(
        move_popover.machine, 0.0, 0.0, speed=None
    )


@pytest.mark.ui
def test_move_to_wcs_zero_uses_speed_getter(move_popover):
    move_popover.set_speed_getter(lambda: 4500)

    move_popover._on_move_to_wcs_zero(None)

    move_popover.machine_cmd.move_to.assert_called_once_with(
        move_popover.machine, 0.0, 0.0, speed=4500
    )


@pytest.fixture
def real_popover(lite_context, sync_machine):
    """A fully constructed MoveToPopover with real widgets."""
    from rayforge.ui_gtk.machine.move_to_popover import MoveToPopover

    sync_machine.set_axis_extents(400, 300)
    popover = MoveToPopover()
    machine_cmd = MagicMock()
    popover.set_machine(sync_machine, machine_cmd)
    return popover, machine_cmd


@pytest.mark.ui
def test_real_popover_click_issues_move(real_popover):
    popover, machine_cmd = real_popover

    assert popover.move_btn.get_sensitive() is True
    assert popover.center_btn.get_sensitive() is False

    popover.set_get_bounds_callback(lambda: (0.0, 0.0, 400.0, 300.0))
    popover.update_sensitivity()
    assert popover.center_btn.get_sensitive() is True

    popover.center_btn.emit("clicked")

    machine_cmd.move_to.assert_called_once()


@pytest.mark.ui
def test_real_popover_show_refreshes_sensitivity(real_popover):
    from gi.repository import Gtk

    popover, _ = real_popover
    parent = Gtk.Window()
    popover.set_parent(parent)
    parent.present()

    popover.move_btn.set_sensitive(False)
    popover.center_btn.set_sensitive(False)

    popover.popup()

    assert popover.move_btn.get_sensitive() is True
    assert popover.center_btn.get_sensitive() is False
    popover.popdown()


class TestPointerAlignmentShifts:
    """While pointer alignment is on, popover aims are shifted so the
    pointer dot lands on the entered/target position."""

    def _set_rows(self, popover, x, y, z=5.0):
        popover.x_row = MagicMock()
        popover.x_row.get_value_in_base_units.return_value = x
        popover.y_row = MagicMock()
        popover.y_row.get_value_in_base_units.return_value = y
        popover.z_row = MagicMock()
        popover.z_row.get_value_in_base_units.return_value = z

    @pytest.mark.ui
    def test_issue_move_shifted_by_offset(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        self._set_rows(move_popover, 50.0, 60.0)

        move_popover._issue_move()

        # The beam is commanded one offset behind the entered value so
        # the pointer dot lands at (50, 60).
        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, 40.0, 40.0, 5.0, speed=None
        )

    @pytest.mark.ui
    def test_issue_move_unshifted_while_off(self, move_popover):
        head = move_popover.machine.get_default_laser_head()
        assert head is not None
        head.set_pointer_offset(10.0, 20.0)
        head.set_pointer_offset_enabled(True)
        self._set_rows(move_popover, 50.0, 60.0)

        move_popover._issue_move()

        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, 50.0, 60.0, 5.0, speed=None
        )

    @pytest.mark.ui
    def test_prefill_reports_pointer_position(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        move_popover.machine.device_state.machine_pos = (
            100.0,
            120.0,
            8.0,
        )
        self._set_rows(move_popover, 0.0, 0.0, 0.0)

        move_popover._prefill_from_machine()

        # Beam at (100, 120) means the pointer dot is at (110, 140).
        move_popover.x_row.set_value_in_base_units.assert_called_once_with(
            110.0
        )
        move_popover.y_row.set_value_in_base_units.assert_called_once_with(
            140.0
        )
        move_popover.z_row.set_value_in_base_units.assert_called_once_with(8.0)

    @pytest.mark.ui
    def test_prefill_round_trip_does_not_move(self, move_popover):
        """Prefill then issuing the prefilled values is a no-move."""
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        move_popover.machine.device_state.machine_pos = (
            100.0,
            120.0,
            8.0,
        )
        self._set_rows(move_popover, 110.0, 140.0)

        move_popover._issue_move()

        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, 100.0, 120.0, 5.0, speed=None
        )

    @pytest.mark.ui
    def test_prefill_shifts_by_wcs_and_offset(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 5.0, 5.0)
        machine = move_popover.machine
        machine.update_wcs_offset("G54", (10.0, 20.0, 3.0))
        machine.active_wcs = "G54"
        machine.device_state.machine_pos = (110.0, 120.0, 8.0)
        self._set_rows(move_popover, 0.0, 0.0, 0.0)

        move_popover._prefill_from_machine()

        # Plain WCS reading is (100, 100); the pointer adds the offset.
        move_popover.x_row.set_value_in_base_units.assert_called_once_with(
            105.0
        )
        move_popover.y_row.set_value_in_base_units.assert_called_once_with(
            105.0
        )

    @pytest.mark.ui
    def test_corner_shortcut_shifted(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)

        move_popover._on_move_to_position(None, "ll")

        # Unshifted the ll corner commands (100, 50); the shift moves
        # the beam one offset behind so the pointer marks the corner.
        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, 90.0, 30.0, speed=None
        )

    @pytest.mark.ui
    def test_wcs_zero_shifted(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        move_popover.machine.update_wcs_offset("G54", (50.0, 50.0, 0.0))

        move_popover._on_move_to_wcs_zero(None)

        # The pointer lands on the origin, so the beam goes to
        # (0, 0) - offset in WCS terms.
        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, -10.0, -20.0, speed=None
        )

    @pytest.mark.ui
    def test_move_clamps_and_warns_out_of_travel(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        self._set_rows(move_popover, 5.0, 5.0)

        move_popover._issue_move()

        # Shifted target (-5, -15) is off the bed; the beam is clamped
        # to (0, 0).
        move_popover.machine_cmd.move_to.assert_called_once_with(
            move_popover.machine, 0.0, 0.0, 5.0, speed=None
        )
        move_popover.travel_warning_label.set_visible.assert_called_with(True)

    @pytest.mark.ui
    def test_move_no_warning_within_travel(self, move_popover):
        enable_pointer_alignment(move_popover.machine, 10.0, 20.0)
        self._set_rows(move_popover, 50.0, 60.0)

        move_popover._issue_move()

        move_popover.travel_warning_label.set_visible.assert_called_once_with(
            False
        )


@pytest.fixture
def real_alignment_popover(lite_context, sync_machine):
    """A fully constructed MoveToPopover with real widgets."""
    from rayforge.ui_gtk.machine.move_to_popover import MoveToPopover

    sync_machine.set_axis_extents(400, 300)
    popover = MoveToPopover()
    machine_cmd = MagicMock()
    popover.set_machine(sync_machine, machine_cmd)
    return popover, machine_cmd


class TestPointerAlignmentSwitch:
    """The alignment switch syncs bidirectionally with the machine."""

    @pytest.mark.ui
    def test_switch_insensitive_without_offset(self, real_alignment_popover):
        popover, _ = real_alignment_popover

        assert popover.alignment_row.get_sensitive() is False
        assert popover.alignment_row.get_active() is False
        assert popover.alignment_row.get_subtitle() == (
            "Requires a pointer offset on the laser head"
        )

    @pytest.mark.ui
    def test_switch_enables_with_offset(self, real_alignment_popover):
        popover, _ = real_alignment_popover
        head = popover.machine.get_default_laser_head()
        assert head is not None
        head.set_pointer_offset(10.0, 20.0)
        head.set_pointer_offset_enabled(True)

        popover._update_bounds_and_axes()

        assert popover.alignment_row.get_sensitive() is True

    @pytest.mark.ui
    def test_switch_toggles_machine_state(self, real_alignment_popover):
        popover, _ = real_alignment_popover
        enable_pointer_alignment(popover.machine)

        popover.alignment_row.set_active(True)

        assert popover.machine.pointer_alignment_enabled is True

        popover.alignment_row.set_active(False)

        assert popover.machine.pointer_alignment_enabled is False

    @pytest.mark.ui
    def test_switch_syncs_from_machine(self, real_alignment_popover):
        popover, _ = real_alignment_popover
        enable_pointer_alignment(popover.machine)

        popover.machine.set_pointer_alignment(False)

        assert popover.alignment_row.get_active() is False

        popover.machine.set_pointer_alignment(True)

        assert popover.alignment_row.get_active() is True

    @pytest.mark.ui
    def test_toggle_keeps_entered_values(self, real_alignment_popover):
        """Toggling alignment must not touch the coordinate rows, so
        the user can move back to a previously entered position."""
        popover, _ = real_alignment_popover
        head = popover.machine.get_default_laser_head()
        assert head is not None
        head.set_pointer_offset(10.0, 20.0)
        head.set_pointer_offset_enabled(True)
        popover.machine.device_state.machine_pos = (100.0, 100.0, 0.0)

        popover.x_row.set_value_in_base_units(50.0)
        popover.y_row.set_value_in_base_units(60.0)

        popover.machine.set_pointer_alignment(True)

        assert popover.x_row.get_value_in_base_units() == 50.0
        assert popover.y_row.get_value_in_base_units() == 60.0

    @pytest.mark.ui
    def test_alignment_resets_when_offset_removed(
        self, real_alignment_popover
    ):
        popover, _ = real_alignment_popover
        head = popover.machine.get_default_laser_head()
        assert head is not None
        enable_pointer_alignment(popover.machine)

        head.set_pointer_offset_enabled(False)

        assert popover.machine.pointer_alignment_enabled is False
        assert popover.alignment_row.get_active() is False
        assert popover.alignment_row.get_sensitive() is False
