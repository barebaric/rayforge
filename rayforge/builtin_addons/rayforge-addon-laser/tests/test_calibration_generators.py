"""Unit tests for the pure parts of the calibration test generators."""

from types import SimpleNamespace
from typing import Any

import pytest
from laser_essentials.calibration import (
    GENERATORS,
    CalibrationLayout,
    CalibrationRecord,
    CalibrationState,
    CellEntry,
    CommandEntry,
    FocusMode,
    VectorShape,
)
from laser_essentials.calibration.builder import (
    CalibrationBuilder,
    bounds_of,
)
from laser_essentials.calibration.focus_test import (
    available_modes,
    focus_commands,
    focus_offsets,
    format_offset,
)
from laser_essentials.calibration.generator import (
    CalibrationGenerator,
    _update_overscan,
)
from laser_essentials.calibration.interval_test import (
    interval_values,
    lines_per_inch,
)
from laser_essentials.calibration.layout import (
    FilledRect,
    ProceduralShape,
    line,
    row_positions,
    text,
)
from laser_essentials.calibration.lookup import (
    _legacy_record,
    record_for_items,
)
from laser_essentials.calibration.material_test_grid import (
    NO_PRESET,
    OFFSET_TEST_LINE_INTERVAL,
    MaterialTestGridGenerator,
    params_from_step,
)
from laser_essentials.material_test_helpers import (
    draw_material_test_preview,
    get_material_test_proportional_size,
)
from laser_essentials.steps import MaterialTestStep

from rayforge.core.step_registry import step_registry
from rayforge.core.varset import TupleVar
from rayforge.core.workpiece import WorkPiece
from rayforge.image.procedural import ProceduralImporter
from rayforge.image.svg.importer import SvgImporter


def _interval(**kwargs):
    params = {"min_interval": 0.05, "max_interval": 0.25, "count": 5}
    params.update(kwargs)
    return params


def _focus(**kwargs):
    params = {"start_offset": -2.0, "step": 0.5, "count": 9}
    params.update(kwargs)
    return params


class TestIntervalValues:
    def test_linear_including_both_ends(self):
        assert interval_values(_interval()) == pytest.approx(
            [0.05, 0.10, 0.15, 0.20, 0.25]
        )

    def test_rejects_fewer_than_two_cells(self):
        with pytest.raises(ValueError):
            interval_values(_interval(count=1))

    @pytest.mark.parametrize("low,high", [(0.2, 0.1), (0.0, 0.1)])
    def test_rejects_bad_range(self, low, high):
        with pytest.raises(ValueError):
            interval_values(_interval(min_interval=low, max_interval=high))

    def test_lines_per_inch(self):
        assert lines_per_inch(0.1) == pytest.approx(254.0)


class TestRowPositions:
    def test_cells_are_spaced_left_to_right(self):
        assert row_positions(3, 10.0, 2.0) == pytest.approx([0.0, 12.0, 24.0])


class TestFocusOffsets:
    def test_offsets(self):
        params = _focus(start_offset=-1.0, step=0.5, count=5)
        assert focus_offsets(params) == pytest.approx(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )

    def test_rejects_zero_step(self):
        with pytest.raises(ValueError):
            focus_offsets(_focus(step=0.0))

    def test_rejects_fewer_than_two_lines(self):
        with pytest.raises(ValueError):
            focus_offsets(_focus(count=1))

    def test_rejects_unreasonable_range(self):
        with pytest.raises(ValueError):
            focus_offsets(_focus(start_offset=-8.0, step=1.0, count=20))

    @pytest.mark.parametrize(
        "value,text", [(-2.0, "-2.0"), (0.0, "0.0"), (0.5, "+0.5")]
    )
    def test_format_offset(self, value, text):
        assert format_offset(value) == text


class TestFocusCommands:
    def test_ramp_needs_no_commands(self):
        assert focus_commands(FocusMode.RAMP, _focus(), False) == {}

    def test_manual_pauses_before_every_line(self):
        params = _focus(start_offset=-0.5, step=0.5, count=3)
        commands = focus_commands(FocusMode.MANUAL, params, False)
        assert sorted(commands) == [0, 1, 2]
        assert all(c.lines == ["M0"] for c in commands.values())
        assert "-0.5" in commands[0].name
        assert "0.5" in commands[1].name

    def test_z_axis_moves_relatively_and_returns(self):
        params = _focus(start_offset=-1.0, step=0.5, count=3)
        commands = focus_commands(FocusMode.Z_AXIS, params, False)
        assert sorted(commands) == [0, 1, 2, 3]
        assert commands[0].lines == ["G91", "G0 Z-1", "G90"]
        assert commands[1].lines == ["G91", "G0 Z0.5", "G90"]
        assert commands[2].lines == ["G91", "G0 Z0.5", "G90"]
        assert commands[3].lines == ["G91", "G0 Z0", "G90"]

    def test_z_axis_return_balances_moves(self):
        params = _focus(start_offset=0.5, step=0.25, count=4)
        commands = focus_commands(FocusMode.Z_AXIS, params, False)
        total = sum(float(c.lines[1].split("Z")[1]) for c in commands.values())
        assert total == pytest.approx(0.0)

    def test_z_axis_respects_reversed_axis(self):
        params = _focus(start_offset=-1.0, step=0.5, count=2)
        commands = focus_commands(FocusMode.Z_AXIS, params, True)
        assert commands[0].lines == ["G91", "G0 Z1", "G90"]
        assert commands[1].lines == ["G91", "G0 Z-0.5", "G90"]

    def test_help_text_per_mode(self):
        texts = {mode.help_text for mode in FocusMode}
        assert len(texts) == 3


class TestLayout:
    def test_cells_and_commands_keep_their_order(self):
        step = MaterialTestStep()
        cell = CellEntry("a", VectorShape(line(0, 0, 1, 0)), step)
        command = CommandEntry("pause", ["M0"])
        layout = CalibrationLayout([command, cell])
        assert layout.cells == [cell]
        assert layout.commands == [command]
        assert layout.entries == [command, cell]


class TestRegistry:
    def test_all_three_tests_are_registered(self):
        assert sorted(GENERATORS) == [
            "focus_test",
            "interval_test",
            "material_test_grid",
        ]
        for generator_id, generator in GENERATORS.items():
            assert generator.id == generator_id
            assert generator.title

    def test_not_available_without_machine(self):
        context: Any = SimpleNamespace(machine=None)
        for generator in GENERATORS.values():
            assert not generator.is_available(context)


class TestRecord:
    def test_round_trip(self):
        record = CalibrationRecord(
            generator_id="material_test_grid",
            layer_uid="layer",
            state=CalibrationState(
                params={"grid_dimensions": (3, 4), "spacing": 1.0},
                workpiece_uids=("wp",),
                step_uids=("s1", "s2"),
                asset_uids=("a",),
            ),
            name="Material Test Grid",
        )
        data = record.to_dict()
        assert data["type"] == "calibration_test"
        assert data["params"]["grid_dimensions"] == [3, 4]
        copy = CalibrationRecord.from_dict(data)
        assert copy.uid == record.uid
        assert copy.name == record.name
        assert copy.layer_uid == "layer"
        assert copy.state.workpiece_uids == ("wp",)
        assert copy.state.step_uids == ("s1", "s2")
        assert copy.state.asset_uids == ("a",)
        assert copy.params == {"grid_dimensions": [3, 4], "spacing": 1.0}

    def test_is_hidden_and_owns_its_workpieces(self):
        record = CalibrationRecord(
            "interval_test",
            "layer",
            CalibrationState({}, workpiece_uids=("a", "b")),
        )
        assert record.hidden
        assert record.owns("a")
        assert not record.owns("c")
        assert record.get_thumbnail(64) is None

    def test_set_state_notifies(self):
        record = CalibrationRecord(
            "interval_test", "layer", CalibrationState({"count": 2})
        )
        seen = []
        record.updated.connect(lambda sender: seen.append(sender), weak=False)
        record.set_state(CalibrationState({"count": 3}))
        assert record.params == {"count": 3}
        assert seen == [record]

    def test_params_are_a_copy(self):
        record = CalibrationRecord(
            "interval_test", "layer", CalibrationState({"count": 2})
        )
        record.params["count"] = 9
        assert record.params == {"count": 2}


class TestMaterialTestGrid:
    generator = MaterialTestGridGenerator()

    def _context(self, max_speed=5000.0) -> Any:
        return SimpleNamespace(
            machine=SimpleNamespace(max_cut_speed=max_speed)
        )

    def test_defaults_clamp_speed_to_machine(self):
        defaults = self.generator.defaults(self._context(300.0))
        assert defaults["speed_range"] == (100.0, 300.0)
        assert defaults["preset"] == NO_PRESET

    def test_speed_range_bound_follows_machine(self):
        varset = self.generator.varset(self._context(8000.0))
        speed_range = varset["speed_range"]
        assert isinstance(speed_range, TupleVar)
        assert speed_range.max_val == 8000.0

    def test_preset_fills_in_values_and_clamps(self):
        params = self.generator.defaults(self._context())
        params["preset"] = "Diode Engrave"
        result = self.generator.apply_change(
            params,
            {"preset"},
            self._context(),
        )
        assert result["test_type"] == "Engrave"
        assert result["power_range"] == (10.0, 100.0)
        assert result["speed_range"] == (1000.0, 5000.0)

    def test_preset_only_applies_when_changed(self):
        params = self.generator.defaults(self._context())
        params["preset"] = "CO2 Cut"
        params["power_range"] = (1.0, 2.0)
        result = self.generator.apply_change(
            params,
            {"spacing"},
            self._context(),
        )
        assert result["power_range"] == (1.0, 2.0)

    def test_speed_vs_offset_switches_to_engrave(self):
        params = self.generator.defaults(self._context())
        params["grid_mode"] = "Speed vs Offset"
        result = self.generator.apply_change(
            params,
            {"grid_mode"},
            self._context(),
        )
        assert result["test_type"] == "Engrave"
        assert result["line_interval_mm"] == OFFSET_TEST_LINE_INTERVAL

    def test_dimension_labels_follow_mode(self):
        labels, _subtitles = self.generator.row_labels(
            {"grid_mode": "Speed vs Passes"}
        )["grid_dimensions"]
        assert labels == ("Columns (Speed Steps)", "Rows (Passes Steps)")

    def test_params_from_step(self):
        step = MaterialTestStep()
        step.grid_dimensions = (7, 3)
        params = params_from_step(step)
        assert params["grid_dimensions"] == (7, 3)
        assert params["line_interval_mm"] == 0.0

    def test_help_text(self):
        assert self.generator.help_text({})

    def test_counts_are_whole_numbers(self):
        params = self.generator.defaults(self._context())
        params["grid_dimensions"] = [3.0, 4.0]
        params["passes_range"] = (1.0, 5.0)
        grid = self.generator.grid_params(params)
        assert grid["grid_dimensions"] == (3, 4)
        assert grid["passes_range"] == (1, 5)
        assert all(isinstance(v, int) for v in grid["passes_range"])

    def test_rejects_too_small_grid(self):
        params = self.generator.defaults(self._context())
        params["grid_dimensions"] = (1, 4)
        with pytest.raises(ValueError):
            self.generator.grid_params(params)


class TestHelpers:
    def test_text_shrinks_to_fit(self):
        wide = text("0.25 mm", 2.5, 0.0, 0.0)
        narrow = text("0.25 mm", 2.5, 0.0, 0.0, max_width=2.0)
        min_x, _min_y, max_x, _max_y = narrow.rect()
        assert max_x - min_x == pytest.approx(2.0)
        assert wide.rect()[2] - wide.rect()[0] > 2.0

    def test_base_generator_has_no_help(self):
        generator = GENERATORS["interval_test"]
        assert CalibrationGenerator.help_text(generator, {}) == ""

    def test_engrave_step_without_machine_keeps_overscan(self):
        step = MaterialTestStep()
        step.per_workpiece_transformers_dicts = [
            {"name": "OverscanTransformer", "auto": True, "distance_mm": 1.0}
        ]
        context: Any = SimpleNamespace(machine=None)
        _update_overscan(context, step)
        assert step.per_workpiece_transformers_dicts[0]["distance_mm"] == 1.0

    def test_focus_modes_without_machine(self):
        context: Any = SimpleNamespace(machine=None)
        assert available_modes(context) == [FocusMode.RAMP]


class TestBuilder:
    def test_bounds_fall_back_to_the_box_without_geometry(self):
        wp = WorkPiece("empty")
        wp.set_size(4.0, 3.0)
        wp.pos = (10.0, 20.0)
        assert bounds_of([wp]) == pytest.approx((10.0, 20.0, 14.0, 23.0))

    def test_command_needs_the_automation_addon(self, monkeypatch):
        monkeypatch.setattr(step_registry, "get", lambda name: None)
        context: Any = SimpleNamespace(machine=None)
        builder = CalibrationBuilder(context)
        layout = CalibrationLayout([CommandEntry("pause", ["M0"])])
        with pytest.raises(ValueError):
            builder.build(layout)

    @pytest.mark.parametrize(
        "shape",
        [
            VectorShape(line(0, 0, 1, 0)),
            FilledRect(0, 0, 1, 1),
        ],
    )
    def test_failed_svg_import_is_an_error(self, monkeypatch, shape):
        monkeypatch.setattr(
            SvgImporter, "get_doc_items", lambda self, spec: None
        )
        context: Any = SimpleNamespace(machine=None)
        builder = CalibrationBuilder(context)
        layout = CalibrationLayout([CellEntry("a", shape, MaterialTestStep())])
        with pytest.raises(RuntimeError):
            builder.build(layout)

    def test_failed_procedural_import_is_an_error(self, monkeypatch):
        monkeypatch.setattr(
            ProceduralImporter, "get_doc_items", lambda self, spec: None
        )
        shape = ProceduralShape(
            draw_function=draw_material_test_preview,
            size_function=get_material_test_proportional_size,
            params={},
        )
        context: Any = SimpleNamespace(machine=None)
        builder = CalibrationBuilder(context)
        layout = CalibrationLayout([CellEntry("a", shape, MaterialTestStep())])
        with pytest.raises(RuntimeError):
            builder.build(layout)


class TestLookup:
    def test_items_outside_a_document(self):
        assert record_for_items([WorkPiece("loose")]) is None

    def test_workpiece_without_layer(self):
        assert _legacy_record(WorkPiece("loose")) is None
