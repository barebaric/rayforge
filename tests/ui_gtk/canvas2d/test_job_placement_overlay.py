from unittest.mock import MagicMock

import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.transport import TransportStatus
from rayforge.ui_gtk.canvas2d import surface as surface_module
from rayforge.ui_gtk.canvas2d.surface import WorkSurface

DESIGN_RECT = (10.0, 20.0, 30.0, 40.0)


@pytest.fixture
def surface(ui_context_initializer, monkeypatch):
    monkeypatch.setattr(
        surface_module, "job_design_rect", lambda doc: DESIGN_RECT
    )
    machine = Machine(ui_context_initializer)
    machine.set_axis_extents(400, 400)
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.set_active_wcs("G54")
    machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))
    editor = MagicMock()
    editor.doc = Doc()
    return WorkSurface(editor, MagicMock(), machine)


def _placed(surface):
    element = surface.job_placement_element
    tx, ty = element.transform.get_translation()
    return (tx, ty, tx + element.width, ty + element.height), element.anchor


@pytest.mark.ui
def test_absolute_shows_no_outline(surface):
    assert not surface.job_placement_element.visible


@pytest.mark.ui
def test_user_origin_outline_on_wcs_zero(surface):
    surface.doc.set_job_origin(
        JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
    )
    assert surface.job_placement_element.visible
    rect, anchor = _placed(surface)
    assert rect == pytest.approx((90.0, 40.0, 110.0, 60.0))
    assert anchor == pytest.approx((10.0, 10.0))


@pytest.mark.ui
def test_current_position_outline_follows_head(surface):
    machine = surface.machine
    surface.doc.set_job_origin(
        JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.BOTTOM_LEFT)
    )
    assert not surface.job_placement_element.visible

    machine.set_connection_status(TransportStatus.CONNECTED)
    state = DeviceState(status=DeviceStatus.IDLE, machine_pos=(200, 150, 0))
    machine.set_device_state(state)
    surface._on_machine_state_changed(machine, state)
    assert surface.job_placement_element.visible
    rect, anchor = _placed(surface)
    assert rect == pytest.approx((200.0, 150.0, 220.0, 170.0))
    assert anchor == pytest.approx((0.0, 0.0))

    surface.doc.set_job_origin(JobOrigin())
    assert not surface.job_placement_element.visible
