import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.transport import TransportStatus


@pytest.fixture
def rows(sync_machine):
    from rayforge.ui_gtk.machine.job_origin_rows import JobOriginRows

    sync_machine.set_active_wcs("G54")
    rows = JobOriginRows()
    rows.set_machine(sync_machine)
    rows.set_doc(Doc())
    return rows


@pytest.mark.ui
def test_defaults_to_absolute(rows):
    assert rows.start_from_row.get_selected() == 0
    assert rows.anchor_buttons[JobAnchor.BOTTOM_LEFT].get_active()
    assert not rows.anchor_row.get_sensitive()
    assert rows.start_from_row.get_subtitle() == "As placed on the canvas"


@pytest.mark.ui
def test_choosing_mode_is_undoable(rows):
    doc = rows.doc
    rows.start_from_row.set_selected(2)
    assert doc.job_origin.start_from == StartFrom.USER_ORIGIN
    assert rows.anchor_row.get_sensitive()
    assert "G54" in rows.start_from_row.get_subtitle()

    doc.history_manager.undo()
    assert doc.job_origin == JobOrigin()
    assert rows.start_from_row.get_selected() == 0


@pytest.mark.ui
def test_choosing_anchor(rows):
    rows.start_from_row.set_selected(2)
    rows.anchor_buttons[JobAnchor.CENTER].set_active(True)
    assert rows.doc.job_origin == JobOrigin(
        StartFrom.USER_ORIGIN, JobAnchor.CENTER
    )
    assert not rows.anchor_buttons[JobAnchor.BOTTOM_LEFT].get_active()
    assert rows.anchor_row.get_subtitle() == "Center"


@pytest.mark.ui
def test_current_position_subtitle(rows):
    rows.start_from_row.set_selected(1)
    assert rows.start_from_row.get_subtitle() == (
        "Needs a connected, idle machine"
    )
    machine = rows.machine
    machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))
    machine.set_connection_status(TransportStatus.CONNECTED)
    machine.set_device_state(
        DeviceState(status=DeviceStatus.IDLE, machine_pos=(120, 80, 0))
    )
    rows.update()
    assert rows.start_from_row.get_subtitle() == (
        "At the laser head: X 20.00  Y 30.00"
    )
