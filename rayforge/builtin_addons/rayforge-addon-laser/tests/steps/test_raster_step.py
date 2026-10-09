from unittest.mock import MagicMock, patch

import cairo
import cv2
import numpy as np
import pytest
from laser_essentials.steps import EngraveStep
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Geometry
from raygeo.ops import Ops
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.raster import RasterSpec
from raygeo.ops.part import Part
from raygeo.ops.types import CommandCategory, CommandType

from rayforge.core.step_registry import step_registry
from rayforge.core.varset import LabeledChoiceVar
from rayforge.core.workpiece import WorkPiece
from rayforge.image.adjust import ImageAdjustments
from rayforge.image.dither import DitherAlgorithm
from rayforge.pipeline.transformer import OpsTransformer
from rayforge.pipeline.transformer.registry import transformer_registry


@pytest.fixture
def mock_context():
    context = MagicMock()
    machine = MagicMock()
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    machine.acceleration = 3000
    default_head = MagicMock()
    default_head.uid = "test-laser-uid"
    default_head.spot_size_mm = (0.1, 0.1)
    machine.get_default_laser_head.return_value = default_head
    context.machine = machine
    return context


class TestEngraveStep:
    def test_instantiation(self):
        step = EngraveStep(name="Test")
        assert step.typelabel == "Engrave"

    def test_create(self, mock_context):
        step = EngraveStep.create(mock_context, name="Created")
        assert isinstance(step, EngraveStep)
        assert len(step.per_workpiece_transformers_dicts) == 4
        transformer_names = {
            t.get("name") for t in step.per_workpiece_transformers_dicts
        }
        assert "BidirScanOffsetTransformer" in transformer_names
        assert "CropTransformer" in transformer_names
        assert step.selected_head_uid == "test-laser-uid"

    def test_merge_scanlines_enabled_by_default(self, mock_context):
        """Scanline merging ships as part of Optimize, enabled."""
        step = EngraveStep.create(mock_context, name="Created")
        optimize = next(
            t
            for t in step.per_step_transformers_dicts
            if t.get("name") == "Optimize"
        )
        assert optimize.get("merge_scanlines") is True

        # Docs saved before the merge-in lack the key and default to
        # enabled on load.
        data = step.to_dict()
        for t in data["per_step_transformers_dicts"]:
            t.pop("merge_scanlines", None)
        restored = EngraveStep.from_dict(data)
        optimizer = next(
            OpsTransformer.from_dict(t)
            for t in restored.per_step_transformers_dicts
            if t.get("name") == "Optimize"
        )
        assert optimizer.to_dict()["merge_scanlines"] is True

    def test_serialization_includes_step_type(self):
        step = EngraveStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "EngraveStep"

    def test_registry_create_engrave_step(self, mock_context):
        StepClass = step_registry.get("EngraveStep")
        assert StepClass is not None
        step = StepClass.create(mock_context, name="FromRegistry")
        assert type(step).__name__ == "EngraveStep"

    def test_get_assembler_kwargs(self, machine):
        step = EngraveStep(name="Test")
        workpiece = MagicMock(spec=["size"])
        workpiece.size = (100, 100)
        kwargs = step.get_assembler_kwargs(machine, workpiece)
        assert isinstance(kwargs, dict)
        expected_keys = {
            "mode",
            "line_interval_mm",
            "sample_interval_mm",
            "dot_width_correction_mm",
            "min_power",
            "max_power",
            "step_power",
            "num_power_levels",
            "angle",
            "offset_x_mm",
            "offset_y_mm",
            "scan_mode",
            "cross_hatch",
            "num_depth_levels",
            "z_step_down",
            "angle_increment",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_roundtrip_serialization(self):
        step = EngraveStep(name="Test")
        step.scan_angle = 45.0
        step.depth_mode = "MULTI_PASS"
        step.line_interval_mm = 0.2  # type: ignore[assignment]
        step.dot_width_correction_mm = 0.05  # type: ignore[assignment]
        data = step.to_dict()
        restored = EngraveStep.from_dict(data)
        assert data == restored.to_dict()
        assert restored.dot_width_correction_mm == 0.05

    def test_legacy_power_keys_migrate(self):
        """Old files keyed the raster power range as min_power/max_power.

        Those must load into min_power_level/max_power_level and must not
        pollute extra. The hardware max_power slot is restored to its
        default rather than inheriting the old raster ceiling.
        """
        step = EngraveStep(name="Test")
        data = step.to_dict()
        data["min_power"] = data.pop("min_power_level")
        data["max_power"] = data.pop("max_power_level")
        data["min_power"] = 0.2
        data["max_power"] = 1.0

        restored = EngraveStep.from_dict(data)

        assert restored.min_power_level == 0.2
        assert restored.max_power_level == 1.0
        assert restored.max_power == 1000
        assert "min_power" not in restored.extra
        assert "max_power" not in restored.extra

    def test_from_dict_migrates_legacy_opsproducer_params(self):
        """True legacy files store raster params in
        ``opsproducer_dict.params``; loading must restore them."""
        step = EngraveStep(name="Test")
        data = step.to_dict()
        for key in (
            "scan_angle",
            "depth_mode",
            "invert",
            "auto_levels",
            "black_point",
            "white_point",
            "threshold",
            "line_interval_mm",
            "sample_interval_mm",
            "min_power_level",
            "max_power_level",
            "num_power_levels",
            "scan_mode",
            "cross_hatch",
            "num_depth_levels",
            "z_step_down",
            "angle_increment",
            "dither_algorithm",
        ):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "Rasterizer",
            "params": {
                "direction_degrees": 45.0,
                "scan_mode": "FullSweep",
                "threshold": 100,
                "dither_algorithm": "bayer4",
                "cross_hatch": True,
                "min_power": 0.2,
                "max_power": 0.9,
                "num_depth_levels": 3,
                "num_power_levels": 10,
                "z_step_down": 0.5,
                "invert": True,
                "auto_levels": False,
                "black_point": 20,
                "white_point": 200,
                "angle_increment": 30.0,
                "line_interval_mm": 0.4,
            },
        }

        restored = EngraveStep.from_dict(data)

        assert restored.depth_mode == "CONSTANT_POWER"
        assert restored.scan_angle == 45.0
        assert restored.scan_mode == "FULL_SWEEP"
        assert restored.threshold == 100
        assert restored.dither_algorithm is not None
        assert restored.dither_algorithm.name == "BAYER4"
        assert restored.cross_hatch is True
        assert restored.min_power_level == 0.2
        assert restored.max_power_level == 0.9
        assert restored.num_depth_levels == 3
        assert restored.num_power_levels == 10
        assert restored.z_step_down == 0.5
        assert restored.invert is True
        assert restored.auto_levels is False
        assert restored.black_point == 20
        assert restored.white_point == 200
        assert restored.angle_increment == 30.0
        assert restored.line_interval_mm == 0.4
        assert restored.max_power == 1000

    def test_from_dict_dither_rasterizer_uses_dither_mode(self):
        """The legacy ``DitherRasterizer`` type implies DITHER mode."""
        step = EngraveStep(name="Test")
        data = step.to_dict()
        for key in ("depth_mode", "scan_angle", "threshold"):
            data.pop(key, None)
        data["opsproducer_dict"] = {
            "type": "DitherRasterizer",
            "params": {"threshold": 150},
        }

        restored = EngraveStep.from_dict(data)

        assert restored.depth_mode == "DITHER"
        assert restored.threshold == 150


class TestEngraveComputePayload:
    """Verifies EngraveStep's build_compute_payload (B3)."""

    def test_build_compute_payload_returns_raster_spec(self, machine):
        step = EngraveStep(name="engrave")
        step.min_power_level = 0.1
        step.max_power_level = 0.9
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)

        with patch.object(WorkPiece, "render_to_pixels", return_value=None):
            part, payload = step.build_compute_payload(machine, wp)

        assert isinstance(part, Part)
        assert isinstance(payload, ComputePayload)
        assert isinstance(payload.assembler, Assembler)
        spec = payload.assembler.spec
        assert isinstance(spec, RasterSpec)
        assert spec.min_power == 0.1
        assert spec.max_power == 0.9
        assert spec.mode == "power_modulated"

    def test_assembler_token_params_mirrors_kwargs(self, machine):
        step = EngraveStep(name="engrave")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 10.0)
        token = step.assembler_token_params(machine, wp)
        kwargs = step.get_assembler_kwargs(machine, wp)
        assert token == kwargs

    def test_set_dither_algorithm_accepts_name_and_enum(self):
        step = EngraveStep(name="engrave")
        step.set_dither_algorithm("BAYER4")
        assert step.dither_algorithm is DitherAlgorithm.BAYER4

        step.set_dither_algorithm(DitherAlgorithm.FLOYD_STEINBERG)
        assert step.dither_algorithm is DitherAlgorithm.FLOYD_STEINBERG

        step.set_dither_algorithm(None)
        assert step.dither_algorithm is None

    def test_set_dither_algorithm_empty_name_means_auto(self):
        step = EngraveStep(name="engrave")
        step.set_dither_algorithm("")
        assert step.dither_algorithm is None


class TestEngraveCheck:
    """Verifies EngraveStep.check() depth-mode capability warning."""

    def _machine(self, supports_multi_depth: bool):
        machine = MagicMock()
        machine.max_cut_speed = 5000
        machine.max_travel_speed = 10000
        machine.supports_travel_speed.return_value = False
        machine.supports_multi_depth_raster.return_value = supports_multi_depth
        return machine

    def test_no_warning_when_supported(self):
        step = EngraveStep(name="engrave")
        step.depth_mode = "MULTI_PASS"
        machine = self._machine(supports_multi_depth=True)
        assert step.check(machine) == []

    def test_no_warning_for_other_modes(self):
        step = EngraveStep(name="engrave")
        step.depth_mode = "POWER_MODULATION"
        machine = self._machine(supports_multi_depth=False)
        assert step.check(machine) == []

    def test_warning_on_unsupported_multi_pass(self):
        step = EngraveStep(name="engrave")
        step.depth_mode = "MULTI_PASS"
        machine = self._machine(supports_multi_depth=False)
        warnings = step.check(machine)
        assert len(warnings) == 1
        assert "Multiple Depths" in warnings[0]

    def test_no_warning_without_machine(self):
        step = EngraveStep(name="engrave")
        step.depth_mode = "MULTI_PASS"
        assert step.check(None) == []


def _stock_rect(x, y, width, height):
    geo = Geometry()
    geo.move_to(x, y)
    geo.line_to(x + width, y)
    geo.line_to(x + width, y + height)
    geo.line_to(x, y + height)
    geo.close_path()
    return geo


class TestEngraveCropToStock:
    """Verifies that the Engrave step's Crop to Stock transformer
    actually crops engrave toolpath to the stock boundary, for every
    movement type (scan lines, travel moves, cut lines) and geometry
    type (lines, beziers, arcs)."""

    @staticmethod
    def _crop_only_step(mock_context):
        """A created EngraveStep whose per-workpiece transformer chain
        is reduced to an enabled CropTransformer."""
        step = EngraveStep.create(mock_context, name="Engrave")
        step.per_workpiece_transformers_dicts = [
            t
            for t in step.per_workpiece_transformers_dicts
            if t.get("name") == "CropTransformer"
        ]
        for t in step.per_workpiece_transformers_dicts:
            t["enabled"] = True
        return step

    @staticmethod
    def _apply(step, ops, stock_geometries):
        """Instantiate the step's enabled per-workpiece transformers
        and apply them the way IntentBuilder does."""
        workpiece = WorkPiece(name="wp")
        workpiece.set_size(1.0, 1.0)
        specs = []
        for t_dict in step.per_workpiece_transformers_dicts:
            if not t_dict.get("enabled", True):
                continue
            cls = transformer_registry.get(t_dict["name"])
            assert cls is not None
            transformer = cls.from_dict(t_dict)
            specs.append(
                transformer.to_spec(workpiece, stock_geometries, None)
            )
        Ops.apply_transformers(ops, specs, progress_cb=None)

    @staticmethod
    def _moving_endpoints(ops):
        for i in range(ops.len()):
            if ops.category(i) == CommandCategory.MOVING:
                yield ops.endpoint(i)

    @staticmethod
    def _commands_of_type(ops, command_type):
        return [
            i for i in range(ops.len()) if ops.command_type(i) == command_type
        ]

    @staticmethod
    def _snapshot(ops):
        items = []
        for i in range(ops.len()):
            item = [ops.command_type(i), ops.endpoint(i)]
            if ops.is_scanline(i):
                item.append(list(ops.scanline_data(i)))
            items.append(item)
        return items

    def test_crop_disabled_by_default(self, mock_context):
        step = EngraveStep.create(mock_context, name="Engrave")
        crop = next(
            t
            for t in step.per_workpiece_transformers_dicts
            if t.get("name") == "CropTransformer"
        )
        assert crop["enabled"] is False

    def test_disabled_crop_does_not_change_ops(self, mock_context):
        """Applying the step's default chain with the crop toggle off
        must produce the same output as a chain without the crop
        transformer at all."""
        stock = [_stock_rect(0.3, 0.0, 0.4, 1.0)]
        with_crop = EngraveStep.create(mock_context, name="Engrave")
        without_crop = EngraveStep.create(mock_context, name="Engrave")
        without_crop.per_workpiece_transformers_dicts = [
            t
            for t in without_crop.per_workpiece_transformers_dicts
            if t.get("name") != "CropTransformer"
        ]

        snapshots = []
        for step in (with_crop, without_crop):
            ops = Ops()
            ops.move_to(0.0, 0.5)
            ops.scan_to(1.0, 0.5, 0.0, bytes(range(64)))
            self._apply(step, ops, stock)
            snapshots.append(self._snapshot(ops))

        assert snapshots[0] == snapshots[1]

    def test_enabled_crop_trims_scanline_to_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.3, 0.0, 0.4, 1.0)]

        ops = Ops()
        ops.move_to(0.0, 0.5)
        ops.scan_to(1.0, 0.5, 0.0, bytes(range(256)))
        self._apply(step, ops, stock)

        scanlines = self._commands_of_type(ops, CommandType.SCAN_LINE)
        assert len(scanlines) == 1
        endpoints = list(self._moving_endpoints(ops))
        assert len(endpoints) == 2
        for endpoint in endpoints:
            assert 0.3 - 1e-6 <= endpoint[0] <= 0.7 + 1e-6
            assert endpoint[1] == pytest.approx(0.5, abs=1e-6)
        assert endpoints[1][0] == pytest.approx(0.7, abs=1e-6)

        power = list(ops.scanline_data(scanlines[0]))
        assert len(power) == pytest.approx(102, abs=3)
        assert power[0] == pytest.approx(77, abs=3)
        assert power[-1] == pytest.approx(178, abs=3)

    def test_enabled_crop_removes_scanline_outside_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.0, 0.0, 1.0, 1.0)]

        ops = Ops()
        ops.move_to(1.2, 0.5)
        ops.scan_to(1.9, 0.5, 0.0, bytes([128] * 16))
        self._apply(step, ops, stock)

        assert list(self._moving_endpoints(ops)) == []

    def test_enabled_crop_keeps_scanline_inside_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.0, 0.0, 1.0, 1.0)]

        ops = Ops()
        ops.move_to(0.4, 0.5)
        ops.scan_to(0.6, 0.5, 0.0, bytes([9] * 8))
        self._apply(step, ops, stock)

        scanlines = self._commands_of_type(ops, CommandType.SCAN_LINE)
        assert len(scanlines) == 1
        assert ops.endpoint(scanlines[0]) == pytest.approx(
            (0.6, 0.5, 0.0), abs=1e-6
        )
        assert list(ops.scanline_data(scanlines[0])) == [9] * 8

    def test_enabled_crop_trims_scanlines_and_cut_lines(self, mock_context):
        """Travel moves (MOVE_TO), scan lines and cut lines (LINE_TO)
        in one ops sequence are all confined to the stock."""
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.3, 0.0, 0.4, 1.0)]

        ops = Ops()
        ops.move_to(0.0, 0.25)
        ops.scan_to(1.0, 0.25, 0.0, bytes([64] * 32))
        ops.move_to(0.0, 0.75)
        ops.line_to(1.0, 0.75)
        self._apply(step, ops, stock)

        scanlines = self._commands_of_type(ops, CommandType.SCAN_LINE)
        cut_lines = self._commands_of_type(ops, CommandType.LINE_TO)
        assert len(scanlines) == 1
        assert len(cut_lines) == 1
        endpoints = list(self._moving_endpoints(ops))
        assert len(endpoints) == 4
        for endpoint in endpoints:
            assert 0.3 - 1e-6 <= endpoint[0] <= 0.7 + 1e-6

    def test_enabled_crop_refits_bezier_crossing_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.3, 0.0, 0.4, 1.0)]

        ops = Ops()
        ops.move_to(0.1, 0.5)
        ops.bezier_to((0.3, 0.3, 0.0), (0.7, 0.7, 0.0), (0.9, 0.5, 0.0))
        self._apply(step, ops, stock)

        endpoints = list(self._moving_endpoints(ops))
        assert endpoints
        for endpoint in endpoints:
            assert 0.3 - 1e-6 <= endpoint[0] <= 0.7 + 1e-6

    def test_enabled_crop_keeps_bezier_inside_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.0, 0.0, 1.0, 1.0)]

        ops = Ops()
        ops.move_to(0.3, 0.5)
        ops.bezier_to((0.4, 0.3, 0.0), (0.6, 0.7, 0.0), (0.7, 0.5, 0.0))
        self._apply(step, ops, stock)

        beziers = self._commands_of_type(ops, CommandType.BEZIER_TO)
        assert len(beziers) == 1
        assert ops.endpoint(beziers[0]) == pytest.approx(
            (0.7, 0.5, 0.0), abs=1e-6
        )

    def test_enabled_crop_refits_arc_crossing_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.3, 0.0, 0.4, 1.0)]

        ops = Ops()
        ops.move_to(0.1, 0.5)
        ops.arc_to(0.9, 0.5, 0.4, 0.0, clockwise=True)
        self._apply(step, ops, stock)

        endpoints = list(self._moving_endpoints(ops))
        assert endpoints
        for endpoint in endpoints:
            assert 0.3 - 1e-6 <= endpoint[0] <= 0.7 + 1e-6

    def test_enabled_crop_keeps_arc_inside_stock(self, mock_context):
        step = self._crop_only_step(mock_context)
        stock = [_stock_rect(0.0, 0.0, 1.0, 1.0)]

        ops = Ops()
        ops.move_to(0.4, 0.5)
        ops.arc_to(0.6, 0.5, 0.1, 0.0, clockwise=True)
        self._apply(step, ops, stock)

        arcs = self._commands_of_type(ops, CommandType.ARC_TO)
        assert len(arcs) == 1
        assert ops.endpoint(arcs[0]) == pytest.approx(
            (0.6, 0.5, 0.0), abs=1e-6
        )


class TestEngraveDitherOptions:
    """Serpentine and halftone settings of the Dither mode."""

    def _var(self, key):
        varset = EngraveStep.recipe_varset()
        return next(v for v in varset.vars if v.key == key)

    def test_defaults(self):
        step = EngraveStep(name="engrave")
        assert step.dither_serpentine is False
        assert step.halftone_cell_mm == 0.5
        assert step.halftone_angle == 45.0

    def test_roundtrip_serialization(self):
        step = EngraveStep(name="engrave")
        step.depth_mode = "DITHER"
        step.dither_algorithm = DitherAlgorithm.HALFTONE
        step.dither_serpentine = True
        step.halftone_cell_mm = 0.8
        step.halftone_angle = 15.0
        data = step.to_dict()
        restored = EngraveStep.from_dict(data)
        assert restored.dither_algorithm is DitherAlgorithm.HALFTONE
        assert restored.dither_serpentine is True
        assert restored.halftone_cell_mm == 0.8
        assert restored.halftone_angle == 15.0
        assert data == restored.to_dict()

    def test_old_documents_get_defaults(self):
        data = EngraveStep(name="engrave").to_dict()
        for key in ("dither_serpentine", "halftone_cell_mm", "halftone_angle"):
            data.pop(key)
        restored = EngraveStep.from_dict(data)
        assert restored.dither_serpentine is False
        assert restored.halftone_cell_mm == 0.5
        assert restored.halftone_angle == 45.0

    def test_new_algorithms_load_by_value(self):
        data = EngraveStep(name="engrave").to_dict()
        data["dither_algorithm"] = "stucki"
        restored = EngraveStep.from_dict(data)
        assert restored.dither_algorithm is DitherAlgorithm.STUCKI

    def test_cache_params_cover_the_new_settings(self):
        step = EngraveStep(name="engrave")
        before = step.get_cache_params()
        for attr, value in (
            ("dither_serpentine", True),
            ("halftone_cell_mm", 1.2),
            ("halftone_angle", 10.0),
        ):
            setattr(step, attr, value)
            after = step.get_cache_params()
            assert after[attr] == value
            assert after != before
            before = after

    def test_varset_lists_every_algorithm(self):
        var = self._var("dither_algorithm")
        assert isinstance(var, LabeledChoiceVar)
        names = [var.get_value_for_display(label) for label in var.choices]
        assert names == [a.name for a in DitherAlgorithm]

    @pytest.mark.parametrize(
        "algo, visible",
        [
            ("FLOYD_STEINBERG", True),
            ("STUCKI", True),
            ("ATKINSON", True),
            ("BAYER4", False),
            ("NEWSPRINT", False),
            ("HALFTONE", False),
        ],
    )
    def test_serpentine_only_for_error_diffusion(self, algo, visible):
        var = self._var("dither_serpentine")
        values = {"depth_mode": "DITHER", "dither_algorithm": algo}
        assert var.is_visible(values) is visible
        values["depth_mode"] = "POWER_MODULATION"
        assert var.is_visible(values) is False

    @pytest.mark.parametrize("key", ["halftone_cell_mm", "halftone_angle"])
    def test_halftone_rows_only_for_halftone(self, key):
        var = self._var(key)
        assert var.is_visible(
            {"depth_mode": "DITHER", "dither_algorithm": "HALFTONE"}
        )
        assert not var.is_visible(
            {"depth_mode": "DITHER", "dither_algorithm": "STUCKI"}
        )
        assert not var.is_visible(
            {"depth_mode": "CONSTANT_POWER", "dither_algorithm": "HALFTONE"}
        )

    def test_apply_import_settings_sets_dither_mode(self):
        step = EngraveStep(name="engrave")
        step.apply_import_settings(
            {
                "depth_mode": "DITHER",
                "dither_algorithm": "HALFTONE",
                "halftone_cell_mm": 0.6,
                "halftone_angle": 22.5,
            }
        )
        assert step.depth_mode == "DITHER"
        assert step.dither_algorithm is DitherAlgorithm.HALFTONE
        assert step.halftone_cell_mm == 0.6
        assert step.halftone_angle == 22.5

    def test_build_raster_part_forwards_dither_options(self, machine):
        step = EngraveStep(name="engrave")
        step.depth_mode = "DITHER"
        step.auto_levels = False
        step.dither_algorithm = DitherAlgorithm.HALFTONE
        step.dither_serpentine = True
        step.halftone_cell_mm = 0.7
        step.halftone_angle = 12.0
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 5.0)
        surface = MagicMock()
        target = "laser_essentials.steps.raster_step.preprocess_raster_image"
        with (
            patch.object(WorkPiece, "render_to_pixels", return_value=surface),
            patch(target, return_value=(None, None)) as preprocess,
        ):
            step.build_compute_payload(machine, wp)

        kwargs = preprocess.call_args.kwargs
        assert kwargs["dither_algorithm"] is DitherAlgorithm.HALFTONE
        assert kwargs["dither_serpentine"] is True
        assert kwargs["halftone_cell_mm"] == 0.7
        assert kwargs["halftone_angle"] == 12.0
        assert kwargs["pixels_per_mm_x"] == pytest.approx(20.0)
        assert kwargs["pixels_per_mm_y"] == pytest.approx(10.0)


class TestEngraveImageAdjustments:
    """Gamma, brightness, contrast and sharpen settings."""

    _KEYS = (
        "brightness",
        "contrast",
        "gamma",
        "sharpen_amount",
        "sharpen_radius_mm",
    )

    def _var(self, key):
        varset = EngraveStep.recipe_varset()
        return next(v for v in varset.vars if v.key == key)

    def test_defaults_are_neutral(self):
        step = EngraveStep(name="engrave")
        assert step.brightness == 0
        assert step.contrast == 0
        assert step.gamma == 1.0
        assert step.sharpen_amount == 0
        assert step.sharpen_radius_mm == 0.2
        assert step.image_adjustments == ImageAdjustments()
        assert step.image_adjustments.is_neutral

    def test_image_adjustments_mirror_the_settings(self):
        step = EngraveStep(name="engrave")
        step.brightness = 10
        step.contrast = -20
        step.gamma = 1.5
        step.sharpen_amount = 80
        step.sharpen_radius_mm = 0.3
        assert step.image_adjustments == ImageAdjustments(
            brightness=10,
            contrast=-20,
            gamma=1.5,
            sharpen_amount=80,
            sharpen_radius_mm=0.3,
        )

    def test_roundtrip_serialization(self):
        step = EngraveStep(name="engrave")
        step.brightness = 12
        step.contrast = 34
        step.gamma = 0.8
        step.sharpen_amount = 150
        step.sharpen_radius_mm = 0.5
        data = step.to_dict()
        restored = EngraveStep.from_dict(data)
        for key in self._KEYS:
            assert getattr(restored, key) == getattr(step, key)
        assert data == restored.to_dict()

    def test_old_documents_get_neutral_adjustments(self):
        data = EngraveStep(name="engrave").to_dict()
        for key in self._KEYS:
            data.pop(key)
        restored = EngraveStep.from_dict(data)
        assert restored.image_adjustments == ImageAdjustments()

    def test_cache_params_cover_the_adjustments(self):
        step = EngraveStep(name="engrave")
        before = step.get_cache_params()
        for key, value in zip(self._KEYS, (5, 5, 1.3, 40, 0.4)):
            setattr(step, key, value)
            after = step.get_cache_params()
            assert after[key] == value
            assert after != before
            before = after

    @pytest.mark.parametrize("key", _KEYS)
    def test_rows_follow_the_levels_modes(self, key):
        var = self._var(key)
        for mode in ("POWER_MODULATION", "MULTI_PASS", "DITHER"):
            assert var.is_visible({"depth_mode": mode})
        assert not var.is_visible({"depth_mode": "CONSTANT_POWER"})

    def test_build_compute_payload_forwards_adjustments(self, machine):
        step = EngraveStep(name="engrave")
        step.depth_mode = "DITHER"
        step.gamma = 1.8
        step.sharpen_amount = 60
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 5.0)
        module = "laser_essentials.steps.raster_step"
        with (
            patch.object(
                WorkPiece, "render_to_pixels", return_value=MagicMock()
            ),
            patch(
                f"{module}.preprocess_raster_image",
                return_value=(None, None),
            ) as preprocess,
            patch(
                f"{module}.compute_raster_auto_levels", return_value=None
            ) as levels,
        ):
            step.build_compute_payload(machine, wp)

        expected = step.image_adjustments
        assert preprocess.call_args.kwargs["adjustments"] == expected
        assert levels.call_args.kwargs["adjustments"] == expected


def _gradient_workpiece(width_mm=10.0, height_mm=5.0):
    """A workpiece whose render is a horizontal black-to-white ramp."""

    def render(width, height):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        gradient = cairo.LinearGradient(0, 0, width, 0)
        gradient.add_color_stop_rgb(0, 0, 0, 0)
        gradient.add_color_stop_rgb(1, 1, 1, 1)
        ctx.set_source(gradient)
        ctx.paint()
        return surface

    wp = WorkPiece(name="wp")
    wp.set_size(width_mm, height_mm)
    return wp, render


class TestProcessedPreview:
    """The bitmap the assembler sees, as a viewable image."""

    def test_dither_preview_is_black_and_white_with_square_pixels(
        self, machine
    ):
        step = EngraveStep(name="engrave")
        step.depth_mode = "DITHER"
        step.auto_levels = False
        wp, render = _gradient_workpiece()
        with patch.object(WorkPiece, "render_to_pixels", side_effect=render):
            preview = step.render_processed_preview(machine, wp)

        assert preview is not None
        assert preview.dtype == np.uint8
        # Spot 0.1 mm: 20 px/mm along X, 10 px/mm along Y; the preview
        # repeats rows so a pixel is square again.
        assert preview.shape == (100, 200)
        assert set(np.unique(preview)) <= {0, 255}
        assert preview[:, :20].mean() < 64
        assert preview[:, -20:].mean() > 192

    def test_power_preview_shows_engraved_darkness(self, machine):
        step = EngraveStep(name="engrave")
        step.depth_mode = "POWER_MODULATION"
        step.auto_levels = False
        wp, render = _gradient_workpiece()
        with patch.object(WorkPiece, "render_to_pixels", side_effect=render):
            plain = step.render_processed_preview(machine, wp)
            step.brightness = 30
            brighter = step.render_processed_preview(machine, wp)

        assert plain is not None and brighter is not None
        assert brighter.astype(int).sum() > plain.astype(int).sum()

    def test_preview_none_without_render(self, machine):
        step = EngraveStep(name="engrave")
        wp = WorkPiece(name="wp")
        wp.set_size(10.0, 5.0)
        with patch.object(WorkPiece, "render_to_pixels", return_value=None):
            assert step.render_processed_preview(machine, wp) is None

    def test_save_processed_preview_writes_png(self, machine, tmp_path):
        step = EngraveStep(name="engrave")
        step.depth_mode = "DITHER"
        wp, render = _gradient_workpiece()
        path = tmp_path / "processed.png"
        with patch.object(WorkPiece, "render_to_pixels", side_effect=render):
            assert step.save_processed_preview(machine, wp, path) is True

        saved = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert saved is not None
        assert saved.shape == (100, 200)
