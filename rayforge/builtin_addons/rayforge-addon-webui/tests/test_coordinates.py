"""
Contract tests for the web UI coordinate module.

These pin the :mod:`webui.coordinates` wrapper against the actual
MachineSpace/MachinePanel implementations across every machine
configuration: all 4 origin corners x reverse_x/reverse_y x all 3
panel orientations. If anyone re-derives conversion math here instead
of delegating to MachinePanel, these tests are the tripwire.
"""

import math

import pytest
from raygeo.ops.axis import Axis
from webui import coordinates

from rayforge.machine.models.machine import JogDirection, Machine, Origin
from rayforge.machine.models.machine_panel import PanelOrientation

ORIGINS = [
    Origin.BOTTOM_LEFT,
    Origin.BOTTOM_RIGHT,
    Origin.TOP_LEFT,
    Origin.TOP_RIGHT,
]
REVERSES = [(False, False), (True, False), (False, True), (True, True)]
ORIENTATIONS = [
    PanelOrientation.NATIVE,
    PanelOrientation.ROTATED_LEFT,
    PanelOrientation.ROTATED_RIGHT,
]

JOG_DIRECTIONS = {
    "+x": JogDirection.EAST,
    "-x": JogDirection.WEST,
    "+y": JogDirection.NORTH,
    "-y": JogDirection.SOUTH,
}


def build_machine(
    context,
    origin=Origin.BOTTOM_LEFT,
    reverse_x=False,
    reverse_y=False,
    orientation=PanelOrientation.NATIVE,
) -> Machine:
    machine = Machine(context)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(reverse_x)
    machine.set_reverse_y_axis(reverse_y)
    machine.set_panel_orientation(orientation)
    return machine


@pytest.fixture
def active_machine(headless_context):
    def _activate(**kwargs) -> Machine:
        machine = build_machine(headless_context, **kwargs)
        headless_context.config.set_machine(machine)
        return machine

    return _activate


@pytest.fixture(
    params=[
        (origin, rx, ry, orientation)
        for origin in ORIGINS
        for rx, ry in REVERSES
        for orientation in ORIENTATIONS
    ],
    ids=lambda cfg: (
        f"{cfg[0].value}-rx{int(cfg[1])}-ry{int(cfg[2])}-" + cfg[3].value
    ),
)
def machine(request, active_machine):
    return active_machine(
        origin=request.param[0],
        reverse_x=request.param[1],
        reverse_y=request.param[2],
        orientation=request.param[3],
    )


class TestViewportContract:
    def test_orientation_and_matrix(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["orientation"] == machine.panel.orientation.value
        matrix = machine.panel.get_world_to_panel_2d()
        assert data["world_to_panel"] == [
            [matrix.get(0, 0), matrix.get(0, 1), matrix.get(0, 2)],
            [matrix.get(1, 0), matrix.get(1, 1), matrix.get(1, 2)],
        ]

    def test_axis_flags_match_panel(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        panel = machine.panel
        assert data["x_axis_right"] == panel.x_axis_right
        assert data["y_axis_down"] == panel.y_axis_down
        assert data["x_axis_negative"] == panel.x_axis_negative
        assert data["y_axis_negative"] == panel.y_axis_negative

    def test_extents_margins_workarea(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        panel = machine.panel
        assert data["extents"] == list(panel.extents)
        assert data["margins"] == list(panel.margins)
        wx, wy, ww, wh = panel.get_workarea_panel_rect()
        assert data["workarea"] == [wx, wy, ww, wh]

    def test_workarea_origin_in_panel(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        space = machine.panel.space
        mx, my = space.get_workarea_origin_in_machine()
        px, py = machine.panel.machine_point_to_panel(mx, my)
        assert data["workarea_origin"] == [px, py]

    def test_soft_limits_match_machine(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        limits = machine.get_soft_limits()
        assert data["soft_limits"] == [float(v) for v in limits]

    def test_soft_limits_panel_inside_extents(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        x, y = data["soft_limits_panel"][:2]
        ext_w, ext_h = data["extents"]
        assert -ext_w - 1 <= x <= ext_w + 1
        assert -ext_h - 1 <= y <= ext_h + 1

    def test_rotary_active_flag(self, headless_context):
        headless_context.config.set_machine(build_machine(headless_context))
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["rotary_active"] is False

    def test_wcs_reported_when_not_workarea_origin(
        self, headless_context, machine
    ):
        machine.active_wcs = "G54"
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["wcs"]["name"] == "G54"
        off_x, off_y, _ = machine.get_active_wcs_offset()
        px, py = machine.panel.machine_point_to_panel(off_x, off_y)
        assert data["wcs"]["offset_panel"] == [px, py]

    def test_wcs_absent_when_workarea_origin(self, headless_context, machine):
        machine.wcs_origin_is_workarea_origin = True
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["wcs"] is None


class TestPointRoundTrips:
    def test_panel_to_machine_round_trip(self, headless_context, machine):
        width, height = machine.panel.extents
        points = [
            (0.0, 0.0),
            (width, height),
            (width / 2, height / 2),
            (13.5, 77.25),
        ]
        for x, y in points:
            mx, my = coordinates.panel_point_to_machine(headless_context, x, y)
            px, py = machine.panel.machine_point_to_panel(mx, my)
            assert px == pytest.approx(x, abs=1e-9)
            assert py == pytest.approx(y, abs=1e-9)

    def test_click_to_machine_matches_panel_matrix(
        self, headless_context, machine
    ):
        width, height = machine.panel.extents
        for x, y in [(10.0, 20.0), (width / 3, height / 4)]:
            expected = machine.panel.panel_point_to_machine(x, y)
            got = coordinates.panel_point_to_machine(headless_context, x, y)
            assert got == expected

    def test_world_to_panel_matches_panel_matrix(
        self, headless_context, machine
    ):
        for x, y in [(0.0, 0.0), (50.0, 30.0), (123.4, 5.6)]:
            expected = machine.panel.get_world_to_panel_2d().transform_point(
                (x, y)
            )
            got = coordinates.world_point_to_panel(headless_context, x, y)
            assert got == expected

    def test_world_panel_machine_chain_consistent(
        self, headless_context, machine
    ):
        for wx, wy in [(10.0, 10.0), (80.0, 42.0)]:
            px, py = coordinates.world_point_to_panel(headless_context, wx, wy)
            mx, my = coordinates.panel_point_to_machine(
                headless_context, px, py
            )
            expected = machine.panel.world_point_to_machine(wx, wy)
            assert mx == pytest.approx(expected[0], abs=1e-9)
            assert my == pytest.approx(expected[1], abs=1e-9)

    def test_beam_position_in_panel(self, headless_context, machine):
        machine.device_state.machine_pos = (42.0, 24.0, 1.0)
        data = coordinates.viewport_contract_dict(headless_context)
        px, py = machine.panel.machine_point_to_panel(42.0, 24.0)
        assert data["beam"] == [px, py]

    def test_beam_position_absent_when_unknown(
        self, headless_context, machine
    ):
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["beam"] is None


class TestItemPlacement:
    def test_world_item_to_machine(self, machine):
        pos = (15.0, 25.0)
        size = (40.0, 60.0)
        expected = machine.panel.space.world_item_to_machine(pos, size)
        got = machine.panel.world_item_to_machine(pos, size)
        assert got == expected

    def test_panel_item_matches_world_item_projection(self, machine):
        pos = (15.0, 25.0)
        size = (40.0, 60.0)
        mx, my = machine.panel.world_item_to_machine(pos, size)
        px, py = machine.panel.machine_point_to_panel(mx, my)
        w, h = machine.panel.extents
        assert 0 <= px <= w + 1e-6
        assert 0 <= py <= h + 1e-6

    def test_work_origin_round_trip_via_panel(self, headless_context, machine):
        space = machine.panel.space
        ox, oy = space.get_workarea_origin_in_machine()
        px, py = machine.panel.machine_point_to_panel(ox, oy)
        mx, my = coordinates.panel_point_to_machine(headless_context, px, py)
        assert mx == pytest.approx(ox, abs=1e-9)
        assert my == pytest.approx(oy, abs=1e-9)


class TestNogoZones:
    def _rect_zone(self, machine):
        from rayforge.machine.models.zone import Zone

        zone = Zone()
        zone.params = {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0}
        machine.add_nogo_zone(zone)
        return zone

    def _polygon_corners(self, polygon):
        flat = polygon
        return [tuple(flat[i : i + 2]) for i in range(0, len(flat), 2)]

    def test_rect_zone_projects_to_panel(self, headless_context, machine):
        zone = self._rect_zone(machine)
        data = coordinates.viewport_contract_dict(headless_context)
        polygon = data["nogo_zones"][0]
        corners = self._polygon_corners(polygon)
        assert len(corners) == 4
        mx, my, mw, mh = (
            zone.params["x"],
            zone.params["y"],
            zone.params["w"],
            zone.params["h"],
        )
        machine_corners = [
            (mx, my),
            (mx + mw, my),
            (mx + mw, my + mh),
            (mx, my + mh),
        ]
        for corner, machine_corner in zip(corners, machine_corners):
            expected = machine.panel.machine_point_to_panel(*machine_corner)
            assert corner == pytest.approx(expected, abs=1e-9)

    def test_cylindrical_zone_projects_to_panel_circle(
        self, headless_context, machine
    ):
        from rayforge.machine.models.zone import Zone, ZoneShape

        zone = Zone()
        zone.shape = ZoneShape.CYLINDER
        zone.params = {"x": 100.0, "y": 80.0, "radius": 12.0}
        machine.add_nogo_zone(zone)
        data = coordinates.viewport_contract_dict(headless_context)
        polygon = data["nogo_zones"][0]
        points = self._polygon_corners(polygon)
        center = machine.panel.machine_point_to_panel(100.0, 80.0)
        for px, py in points:
            distance = math.hypot(px - center[0], py - center[1])
            assert distance == pytest.approx(12.0, abs=1e-6)
        assert points[0] == points[-1]

    def test_disabled_zones_are_omitted(self, headless_context, machine):
        zone = self._rect_zone(machine)
        zone.set_enabled(False)
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["nogo_zones"] == []

    def test_no_zones_yields_empty_list(self, headless_context, machine):
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["nogo_zones"] == []


class TestJogDirections:
    def test_invalid_direction_raises(self, headless_context, machine):
        with pytest.raises(ValueError):
            coordinates.jog_for_panel_direction(
                headless_context, "sideways", 10.0
            )

    def test_z_jog_maps_to_z_axis(self, headless_context, machine):
        deltas = coordinates.jog_for_panel_direction(
            headless_context, "+z", 5.0
        )
        assert deltas == {Axis.Z: 5.0}
        deltas = coordinates.jog_for_panel_direction(
            headless_context, "-z", 5.0
        )
        assert deltas == {Axis.Z: -5.0}

    def test_native_jog_matches_machine_calculate_jog(
        self, headless_context, machine
    ):
        if machine.panel.orientation is not PanelOrientation.NATIVE:
            pytest.skip("machine-level parity only holds for NATIVE panels")
        for direction, jog_dir in JOG_DIRECTIONS.items():
            expected = machine.calculate_jog(jog_dir, 10.0)
            deltas = coordinates.jog_for_panel_direction(
                headless_context, direction, 10
            )
            assert sum(deltas.values()) == pytest.approx(expected)

    def test_jog_signs_for_default_machine(self, headless_context):
        headless_context.config.set_machine(build_machine(headless_context))
        assert coordinates.jog_for_panel_direction(
            headless_context, "+x", 10
        ) == {Axis.X: 10.0}
        assert coordinates.jog_for_panel_direction(
            headless_context, "-x", 10
        ) == {Axis.X: -10.0}
        assert coordinates.jog_for_panel_direction(
            headless_context, "+y", 10
        ) == {Axis.Y: 10.0}
        assert coordinates.jog_for_panel_direction(
            headless_context, "-y", 10
        ) == {Axis.Y: -10.0}

    def test_jog_vector_rotates_with_panel(self, headless_context, machine):
        for direction, jog_dir in JOG_DIRECTIONS.items():
            deltas = coordinates.jog_for_panel_direction(
                headless_context, direction, 10
            )
            expected = machine.panel.calculate_jog(jog_dir, 10.0)
            assert deltas == expected
