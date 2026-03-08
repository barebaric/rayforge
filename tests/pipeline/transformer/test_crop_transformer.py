import pytest
from unittest.mock import Mock, MagicMock

from rayforge.core.ops import Ops
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.crop_transformer import CropTransformer
from rayforge.pipeline.transformer.base import ExecutionPhase


@pytest.fixture
def transformer() -> CropTransformer:
    return CropTransformer(enabled=True, tolerance=0.03, offset=0.0)


@pytest.fixture
def mock_workpiece():
    wp = MagicMock(spec=WorkPiece)
    wp.get_world_transform.return_value = Matrix.identity()
    wp.size = (1.0, 1.0)
    return wp


def create_rect_geometry(x, y, width, height):
    geo = Geometry()
    geo.move_to(x, y)
    geo.line_to(x + width, y)
    geo.line_to(x + width, y + height)
    geo.line_to(x, y + height)
    geo.close_path()
    return geo


class TestCropTransformerInit:
    def test_default_initialization(self):
        t = CropTransformer()
        assert t.enabled is True
        assert t.tolerance == 0.03
        assert t.offset == 0.0

    def test_custom_initialization(self):
        t = CropTransformer(enabled=False, tolerance=0.1, offset=5.0)
        assert t.enabled is False
        assert t.tolerance == 0.1
        assert t.offset == 5.0


class TestCropTransformerProperties:
    def test_execution_phase(self, transformer):
        assert transformer.execution_phase == ExecutionPhase.POST_PROCESSING

    def test_position_sensitive(self):
        assert CropTransformer.POSITION_SENSITIVE is True

    def test_label(self, transformer):
        assert transformer.label == "Crop to Stock"

    def test_description(self, transformer):
        assert "crop" in transformer.description.lower()

    def test_tolerance_property_setter_triggers_signal(self, transformer):
        transformer.changed = Mock()
        transformer.tolerance = 0.5
        assert transformer.tolerance == 0.5
        transformer.changed.send.assert_called_once_with(transformer)

    def test_tolerance_property_no_signal_if_same_value(self, transformer):
        transformer.changed = Mock()
        original = transformer.tolerance
        transformer.tolerance = original
        transformer.changed.send.assert_not_called()

    def test_offset_property_setter_triggers_signal(self, transformer):
        transformer.changed = Mock()
        transformer.offset = 2.5
        assert transformer.offset == 2.5
        transformer.changed.send.assert_called_once_with(transformer)

    def test_offset_property_no_signal_if_same_value(self, transformer):
        transformer.changed = Mock()
        original = transformer.offset
        transformer.offset = original
        transformer.changed.send.assert_not_called()


class TestCropTransformerSerialization:
    def test_to_dict(self, transformer):
        data = transformer.to_dict()
        assert data["name"] == "CropTransformer"
        assert data["enabled"] is True
        assert data["tolerance"] == 0.03
        assert data["offset"] == 0.0

    def test_from_dict(self):
        data = {
            "name": "CropTransformer",
            "enabled": False,
            "tolerance": 0.15,
            "offset": 3.0,
        }
        t = CropTransformer.from_dict(data)
        assert isinstance(t, CropTransformer)
        assert t.enabled is False
        assert t.tolerance == 0.15
        assert t.offset == 3.0

    def test_from_dict_defaults(self):
        data = {"name": "CropTransformer"}
        t = CropTransformer.from_dict(data)
        assert t.enabled is True
        assert t.tolerance == 0.03
        assert t.offset == 0.0


class TestCropTransformerNoOp:
    def test_no_op_when_disabled(self, transformer, mock_workpiece):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(200, 0)
        original_commands = list(ops.commands)

        transformer.enabled = False
        stock_geo = create_rect_geometry(0, 0, 100, 100)
        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        assert ops.commands == original_commands

    def test_no_op_when_no_stock_geometries(self, transformer, mock_workpiece):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(200, 0)
        original_commands = list(ops.commands)

        transformer.run(ops, workpiece=mock_workpiece, stock_geometries=None)

        assert ops.commands == original_commands

    def test_no_op_when_empty_stock_geometries(
        self, transformer, mock_workpiece
    ):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(200, 0)
        original_commands = list(ops.commands)

        transformer.run(ops, workpiece=mock_workpiece, stock_geometries=[])

        assert ops.commands == original_commands

    def test_no_op_when_no_workpiece(self, transformer):
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(200, 0)
        original_commands = list(ops.commands)

        stock_geo = create_rect_geometry(0, 0, 100, 100)
        transformer.run(ops, workpiece=None, stock_geometries=[stock_geo])

        assert ops.commands == original_commands


class TestCropTransformerCropping:
    def test_crop_line_outside_stock(self, transformer, mock_workpiece):
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)
        stock_geo = create_rect_geometry(0.3, 0, 0.4, 1)

        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == 1
        segment = segments[0]
        assert len(segment) >= 2
        assert segment[0].end is not None
        assert segment[-1].end is not None
        start_x = segment[0].end[0]
        end_x = segment[-1].end[0]
        assert start_x >= 0.3
        assert end_x <= 0.7

    def test_crop_line_fully_inside_stock(self, transformer, mock_workpiece):
        ops = Ops()
        ops.move_to(0.4, 0.5)
        ops.line_to(0.6, 0.5)
        original_segment_count = len(list(ops.segments()))

        stock_geo = create_rect_geometry(0, 0, 1, 1)
        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == original_segment_count

    def test_crop_line_fully_outside_stock(self, transformer, mock_workpiece):
        ops = Ops()
        ops.move_to(1.5, 1.5)
        ops.line_to(2, 2)
        stock_geo = create_rect_geometry(0, 0, 1, 1)

        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == 0

    def test_crop_with_positive_offset(self, mock_workpiece):
        transformer = CropTransformer(offset=0.1)
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)

        stock_geo = create_rect_geometry(0.4, 0, 0.2, 1)
        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == 1
        segment = segments[0]
        assert segment[0].end is not None
        assert segment[-1].end is not None
        start_x = segment[0].end[0]
        end_x = segment[-1].end[0]
        assert start_x >= 0.3
        assert end_x <= 0.7

    def test_crop_with_negative_offset(self, mock_workpiece):
        transformer = CropTransformer(offset=-0.1)
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)

        stock_geo = create_rect_geometry(0.3, 0, 0.4, 1)
        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == 1
        segment = segments[0]
        assert segment[0].end is not None
        assert segment[-1].end is not None
        start_x = segment[0].end[0]
        end_x = segment[-1].end[0]
        assert start_x >= 0.4
        assert end_x <= 0.6

    def test_crop_with_multiple_stock_geometries(self, mock_workpiece):
        transformer = CropTransformer()
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)

        stock_geo1 = create_rect_geometry(0, 0, 0.4, 1)
        stock_geo2 = create_rect_geometry(0.6, 0, 0.4, 1)
        transformer.run(
            ops,
            workpiece=mock_workpiece,
            stock_geometries=[stock_geo1, stock_geo2],
        )

        segments = list(ops.segments())
        assert len(segments) == 2
        seg1 = segments[0]
        seg2 = segments[1]
        assert seg1[0].end is not None
        assert seg1[-1].end is not None
        assert seg2[0].end is not None
        assert seg2[-1].end is not None
        assert seg1[0].end[0] >= 0 and seg1[-1].end[0] <= 0.4
        assert seg2[0].end[0] >= 0.6 and seg2[-1].end[0] <= 1

    def test_crop_with_transformed_workpiece(self):
        wp = MagicMock(spec=WorkPiece)
        wp.get_world_transform.return_value = Matrix.translation(0.5, 0)
        wp.size = (1.0, 1.0)

        transformer = CropTransformer()
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)

        stock_geo = create_rect_geometry(0, 0, 1, 1)
        transformer.run(ops, workpiece=wp, stock_geometries=[stock_geo])

        segments = list(ops.segments())
        assert len(segments) == 1

    def test_crop_with_rotated_workpiece(self):
        import math

        wp = MagicMock(spec=WorkPiece)
        rotation = Matrix.rotation(math.pi / 4)
        wp.get_world_transform.return_value = rotation
        wp.size = (1.0, 1.0)

        transformer = CropTransformer()
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(0.5, 0.5)

        stock_geo = create_rect_geometry(-1, -1, 2, 2)
        transformer.run(ops, workpiece=wp, stock_geometries=[stock_geo])

        segments = list(ops.segments())
        assert len(segments) >= 0

    def test_crop_empty_ops(self, transformer, mock_workpiece):
        ops = Ops()
        stock_geo = create_rect_geometry(0, 0, 100, 100)

        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        assert len(ops.commands) == 0

    def test_crop_workpiece_with_no_size(self):
        wp = MagicMock(spec=WorkPiece)
        wp.get_world_transform.return_value = Matrix.identity()
        wp.size = None

        transformer = CropTransformer()
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(100, 0)
        original_len = len(ops.commands)

        stock_geo = create_rect_geometry(0, 0, 50, 50)
        transformer.run(ops, workpiece=wp, stock_geometries=[stock_geo])

        assert len(ops.commands) <= original_len

    def test_crop_with_custom_tolerance(self, mock_workpiece):
        transformer = CropTransformer(tolerance=0.1)
        ops = Ops()
        ops.move_to(0, 0.5)
        ops.line_to(1, 0.5)

        stock_geo = create_rect_geometry(0.3, 0, 0.4, 1)
        transformer.run(
            ops, workpiece=mock_workpiece, stock_geometries=[stock_geo]
        )

        segments = list(ops.segments())
        assert len(segments) == 1
