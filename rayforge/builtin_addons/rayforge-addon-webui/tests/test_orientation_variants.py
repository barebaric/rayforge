"""
Contract tests for the machine-orientation variants used for web UI
browser verification (scripts/gen_oriented_configs.py).

Each variant machine is round-tripped through the same dict the
generator writes (and ``Machine.from_dict`` the way ``rayforge serve``
loads it), then the serialized ``GET /viewport`` contract is asserted
against hand-derived expectations for a 200x200 bed with no margins.
The 48-case matrix in test_coordinates.py pins the wrapper to
MachinePanel; these tests pin the concrete variants to absolute values,
so a presentation regression (e.g. an origin marker on the wrong
corner) cannot slip through both layers.
"""

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from webui import coordinates
from webui.serializers import machine_state

from rayforge.machine.models.machine import Machine

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "gen_oriented_configs.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_oriented_configs", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

KEYS = [variant["key"] for variant in gen.VARIANTS]

MACHINE_POS = {"x": 42.0, "y": 24.0, "z": 1.0}
REVERSED_MACHINE_POS = {"x": -42.0, "y": 24.0, "z": 1.0}

EXPECTED = {
    "bottom_left": {
        "orientation": "native",
        "flags": (False, False, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [0.0, 0.0],
        "beam_panel": [42.0, 24.0],
        "machine_pos": MACHINE_POS,
    },
    "top_left": {
        "orientation": "native",
        "flags": (False, True, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [0.0, 200.0],
        "beam_panel": [42.0, 176.0],
        "machine_pos": MACHINE_POS,
    },
    "top_right": {
        "orientation": "native",
        "flags": (True, True, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [200.0, 200.0],
        "beam_panel": [158.0, 176.0],
        "machine_pos": MACHINE_POS,
    },
    "bottom_right": {
        "orientation": "native",
        "flags": (True, False, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [200.0, 0.0],
        "beam_panel": [158.0, 24.0],
        "machine_pos": MACHINE_POS,
    },
    "top_right_reverse_x": {
        "orientation": "native",
        "flags": (True, True, True, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [200.0, 200.0],
        "beam_panel": [158.0, 176.0],
        "machine_pos": REVERSED_MACHINE_POS,
    },
    "bottom_left_reverse_y": {
        "orientation": "native",
        "flags": (False, False, False, True),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [0.0, 0.0],
        "beam_panel": [42.0, 24.0],
        "machine_pos": {"x": 42.0, "y": -24.0, "z": 1.0},
    },
    "bottom_left_rotated_left": {
        "orientation": "rotated_left",
        "flags": (True, False, False, False),
        "world_to_panel": [[0, -1, 200], [1, 0, 0]],
        "workarea_origin": [200.0, 0.0],
        "beam_panel": [176.0, 42.0],
        "machine_pos": MACHINE_POS,
    },
    "top_left_rotated_right": {
        "orientation": "rotated_right",
        "flags": (True, True, False, False),
        "world_to_panel": [[0, 1, 0], [-1, 0, 200]],
        "workarea_origin": [200.0, 200.0],
        "beam_panel": [176.0, 158.0],
        "machine_pos": MACHINE_POS,
    },
    "bottom_left_workarea_origin": {
        "orientation": "native",
        "flags": (False, False, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [0.0, 0.0],
        "work_origin": [0.0, 0.0],
        "beam_panel": [10.0, 20.0],
        "machine_pos": MACHINE_POS,
        "work_pos": {"x": 10.0, "y": 20.0, "z": 0.0},
    },
    "top_right_workarea_origin": {
        "orientation": "native",
        "flags": (True, True, False, False),
        "world_to_panel": [[1, 0, 0], [0, 1, 0]],
        "workarea_origin": [200.0, 200.0],
        "work_origin": [200.0, 200.0],
        "beam_panel": [190.0, 180.0],
        "machine_pos": MACHINE_POS,
        "work_pos": {"x": 10.0, "y": 20.0, "z": 0.0},
    },
}


def build_variant(headless_context, key: str) -> Machine:
    variant = next(v for v in gen.VARIANTS if v["key"] == key)
    data = gen._machine_dict(headless_context, variant)
    machine = Machine.from_dict(data, context=headless_context)
    headless_context.config.set_machine(machine)
    return machine


@pytest.fixture(params=KEYS)
def variant_machine(request, headless_context):
    return build_variant(headless_context, request.param), request.param


class TestVariantContract:
    def test_orientation_and_flags(self, headless_context, variant_machine):
        _machine, key = variant_machine
        expected = EXPECTED[key]
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["orientation"] == expected["orientation"]
        assert data["extents"] == [200.0, 200.0]
        x_right, y_down, x_neg, y_neg = expected["flags"]
        assert data["x_axis_right"] is x_right
        assert data["y_axis_down"] is y_down
        assert data["x_axis_negative"] is x_neg
        assert data["y_axis_negative"] is y_neg

    def test_world_to_panel_matrix(self, headless_context, variant_machine):
        _, key = variant_machine
        data = coordinates.viewport_contract_dict(headless_context)
        expected = EXPECTED[key]["world_to_panel"]
        assert data["world_to_panel"] == [
            [float(v) for v in row] for row in expected
        ]

    def test_workarea_origin_at_displayed_origin_corner(
        self, headless_context, variant_machine
    ):
        """The workarea origin sits in the displayed origin corner: the
        corner the axis arrows must anchor to."""
        _, key = variant_machine
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["workarea_origin"] == EXPECTED[key]["workarea_origin"]
        x, y = data["workarea_origin"]
        assert data["workarea"] == [0.0, 0.0, 200.0, 200.0]
        corners = {
            (0.0, 0.0),
            (200.0, 0.0),
            (0.0, 200.0),
            (200.0, 200.0),
        }
        assert (x, y) in corners

    def test_work_origin_tracks_wcs(self, headless_context, variant_machine):
        machine, key = variant_machine
        data = coordinates.viewport_contract_dict(headless_context)
        if machine.wcs_origin_is_workarea_origin:
            assert data["wcs"] is None
            assert data["work_origin"] == EXPECTED[key]["work_origin"]
            assert data["work_origin"] == data["workarea_origin"]
        else:
            off_x, off_y, _ = machine.get_active_wcs_offset()
            px, py = machine.panel.machine_point_to_panel(off_x, off_y)
            assert data["work_origin"] == [px, py]

    def test_wcs_marker_moves_with_offset(
        self, headless_context, variant_machine
    ):
        machine, _key = variant_machine
        if machine.wcs_origin_is_workarea_origin:
            pytest.skip("workarea-origin mode has no mutable WCS marker")
        machine.coordinate_systems["G54"].offset = (50.0, 60.0, 0.0)
        data = coordinates.viewport_contract_dict(headless_context)
        px, py = machine.panel.machine_point_to_panel(50.0, 60.0)
        assert data["work_origin"] == [px, py]
        assert data["wcs"]["offset_panel"] == [px, py]


class TestVariantBeamPath:
    """Machine report -> screen: DeviceState positions must land on the
    correct panel location per orientation (unit-level; the placeholder
    driver never moves on its own)."""

    def _report_positions(self, machine, key: str) -> None:
        expected = EXPECTED[key]
        pos = expected["machine_pos"]
        machine.device_state.machine_pos = (pos["x"], pos["y"], pos["z"])
        work = expected.get("work_pos")
        if work:
            machine.device_state.work_pos = (work["x"], work["y"], work["z"])

    def test_machine_pos_beam_in_panel(
        self, headless_context, variant_machine
    ):
        machine, key = variant_machine
        self._report_positions(machine, key)
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["beam"] == EXPECTED[key]["beam_panel"]

    def test_work_pos_beam_in_workarea_origin_mode(
        self, headless_context, variant_machine
    ):
        machine, key = variant_machine
        if not machine.wcs_origin_is_workarea_origin:
            pytest.skip(
                "work_pos only drives the beam in workarea-origin mode"
            )
        self._report_positions(machine, key)
        data = coordinates.viewport_contract_dict(headless_context)
        assert data["beam"] == EXPECTED[key]["beam_panel"]

    def test_machine_state_carries_beam(
        self, headless_context, variant_machine
    ):
        machine, key = variant_machine
        self._report_positions(machine, key)
        state = machine_state(machine)
        assert state["beam_position_panel"] == EXPECTED[key]["beam_panel"]

    def test_machine_state_beam_unknown(self, variant_machine):
        machine, _ = variant_machine
        assert machine_state(machine)["beam_position_panel"] is None


class TestSoftLimitsPanel:
    """Asymmetric soft limits must project to the correct PANEL rect
    (the full-bed case is invariant, so it hides corner mistakes)."""

    @pytest.fixture(params=KEYS)
    def limited(self, request, headless_context):
        machine = build_variant(headless_context, request.param)
        machine._soft_limits = (10.0, 20.0, 150.0, 160.0)
        return (
            machine,
            request.param,
            coordinates.viewport_contract_dict(headless_context)[
                "soft_limits_panel"
            ],
        )

    EXPECTED_SOFT_LIMITS_PANEL: ClassVar[dict[str, list[float]]] = {
        "bottom_left": [10.0, 20.0, 140.0, 140.0],
        "top_left": [10.0, 40.0, 140.0, 140.0],
        "top_right": [50.0, 40.0, 140.0, 140.0],
        "bottom_right": [50.0, 20.0, 140.0, 140.0],
        "top_right_reverse_x": [50.0, 40.0, 140.0, 140.0],
        "bottom_left_reverse_y": [10.0, 20.0, 140.0, 140.0],
        "bottom_left_rotated_left": [40.0, 10.0, 140.0, 140.0],
        "top_left_rotated_right": [40.0, 50.0, 140.0, 140.0],
        "bottom_left_workarea_origin": [10.0, 20.0, 140.0, 140.0],
        "top_right_workarea_origin": [50.0, 40.0, 140.0, 140.0],
    }

    def test_panel_rect(self, limited):
        machine, key, panel_rect = limited
        assert panel_rect == self.EXPECTED_SOFT_LIMITS_PANEL[key]
        x, y, w, h = panel_rect
        ext_w, ext_h = machine.panel.extents
        assert 0 <= x <= ext_w
        assert 0 <= y <= ext_h
        assert x + w <= ext_w + 1e-9
        assert y + h <= ext_h + 1e-9
