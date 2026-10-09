import pytest

from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.job_placement import (
    JobPlacementError,
    compute_job_shift,
    resolve_start_point,
)
from rayforge.machine.models.machine import Origin
from rayforge.machine.transport import TransportStatus

DESIGN_RECT = (10.0, 20.0, 30.0, 40.0)


@pytest.fixture
def machine(sync_machine):
    m = sync_machine
    m.set_axis_extents(400, 400)
    m.set_origin(Origin.BOTTOM_LEFT)
    m.set_active_wcs("G54")
    m.update_wcs_offset("G54", (100.0, 50.0, 0.0))
    return m


def _connect_at(machine, x, y, status=DeviceStatus.IDLE):
    machine.set_connection_status(TransportStatus.CONNECTED)
    machine.set_device_state(
        DeviceState(status=status, machine_pos=(x, y, 0.0))
    )


class TestResolveStartPoint:
    def test_absolute_has_no_start_point(self, machine):
        assert resolve_start_point(machine, StartFrom.ABSOLUTE) is None

    def test_user_origin_is_active_wcs_zero(self, machine):
        point = resolve_start_point(machine, StartFrom.USER_ORIGIN)
        assert point == pytest.approx((100.0, 50.0))

    def test_user_origin_with_other_wcs(self, machine):
        machine.update_wcs_offset("G55", (5.0, 6.0, 0.0))
        machine.set_active_wcs("G55")
        point = resolve_start_point(machine, StartFrom.USER_ORIGIN)
        assert point == pytest.approx((5.0, 6.0))

    def test_user_origin_on_top_left_machine(self, machine):
        machine.set_origin(Origin.TOP_LEFT)
        machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))
        point = resolve_start_point(machine, StartFrom.USER_ORIGIN)
        # 50 mm down from the top edge of a 400 mm bed.
        assert point == pytest.approx((100.0, 350.0))

    def test_user_origin_as_workarea_origin(self, machine):
        machine.set_work_margins(7.0, 0.0, 0.0, 3.0)
        machine.set_wcs_origin_is_workarea_origin(True)
        point = resolve_start_point(machine, StartFrom.USER_ORIGIN)
        assert point == pytest.approx((7.0, 3.0))

    def test_current_position_uses_reported_head(self, machine):
        _connect_at(machine, 120.0, 80.0)
        point = resolve_start_point(machine, StartFrom.CURRENT_POSITION)
        assert point == pytest.approx((120.0, 80.0))

    def test_current_position_on_top_right_machine(self, machine):
        machine.set_origin(Origin.TOP_RIGHT)
        _connect_at(machine, 120.0, 80.0)
        point = resolve_start_point(machine, StartFrom.CURRENT_POSITION)
        assert point == pytest.approx((280.0, 320.0))

    def test_current_position_follows_pointer_when_aligning(self, machine):
        head = machine.get_default_laser_head()
        head.set_pointer_offset_enabled(True)
        head.set_pointer_offset(-12.0, 4.0)
        machine.set_pointer_alignment(True)
        _connect_at(machine, 120.0, 80.0)
        point = resolve_start_point(machine, StartFrom.CURRENT_POSITION)
        assert point == pytest.approx((108.0, 84.0))

    def test_current_position_needs_connection(self, machine):
        machine.set_device_state(
            DeviceState(status=DeviceStatus.IDLE, machine_pos=(1, 2, 0))
        )
        with pytest.raises(JobPlacementError):
            resolve_start_point(machine, StartFrom.CURRENT_POSITION)

    def test_current_position_needs_known_position(self, machine):
        machine.set_connection_status(TransportStatus.CONNECTED)
        machine.set_device_state(DeviceState(status=DeviceStatus.IDLE))
        with pytest.raises(JobPlacementError):
            resolve_start_point(machine, StartFrom.CURRENT_POSITION)

    @pytest.mark.parametrize(
        "status",
        [DeviceStatus.RUN, DeviceStatus.JOG, DeviceStatus.ALARM],
    )
    def test_current_position_needs_idle_machine(self, machine, status):
        _connect_at(machine, 120.0, 80.0, status=status)
        with pytest.raises(JobPlacementError):
            resolve_start_point(machine, StartFrom.CURRENT_POSITION)


class TestComputeJobShift:
    def test_absolute_has_no_shift(self, machine):
        assert compute_job_shift(machine, JobOrigin(), DESIGN_RECT) is None

    def test_user_origin_bottom_left(self, machine):
        origin = JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.BOTTOM_LEFT)
        shift = compute_job_shift(machine, origin, DESIGN_RECT)
        assert shift == pytest.approx((90.0, 30.0))

    def test_current_position_center(self, machine):
        _connect_at(machine, 200.0, 150.0)
        origin = JobOrigin(StartFrom.CURRENT_POSITION, JobAnchor.CENTER)
        shift = compute_job_shift(machine, origin, DESIGN_RECT)
        assert shift == pytest.approx((180.0, 120.0))

    def test_empty_design_has_no_shift(self, machine):
        origin = JobOrigin(StartFrom.USER_ORIGIN, JobAnchor.CENTER)
        assert compute_job_shift(machine, origin, None) is None
