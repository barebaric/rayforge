from unittest.mock import MagicMock

import pytest
from raygeo.geo import Geometry, Matrix

from rayforge.core.group import Group
from rayforge.core.layer import Layer
from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import LayerImportMode, PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.assembler import ItemAssembler
from rayforge.image.structures import LayoutItem


@pytest.fixture
def assembler():
    return ItemAssembler()


@pytest.fixture
def source():
    src = MagicMock(spec=SourceAsset)
    src.uid = "test-uid"
    return src


@pytest.fixture
def spec():
    return PassthroughSpec()


@pytest.fixture
def mock_geo():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 1)
    return geo


def test_single_item_creation(assembler, source, spec, mock_geo):
    """
    Verifies that a single layout item creates a WorkPiece.
    """
    plan = [
        LayoutItem(
            layer_id=None,
            layer_name=None,
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(0, 0, 100, 100),
        )
    ]
    geometries: dict[str | None, Geometry] = {"some_layer": mock_geo}

    items = assembler.create_items(source, plan, spec, "TestFile", geometries)

    assert len(items) == 1
    wp = items[0]
    assert isinstance(wp, WorkPiece)
    assert wp.name == "TestFile"

    if isinstance(wp, WorkPiece) and wp.source_segment:
        assert wp.source_segment.source_asset_uid == "test-uid"
        assert wp.source_segment.crop_window_px == (0, 0, 100, 100)
        assert wp.source_segment.pristine_geometry is not None
        assert wp.source_segment.pristine_geometry is not mock_geo


def test_split_layers_creation(assembler, source, mock_geo):
    """
    Verifies that layout items with layer_ids creates Layer objects
    containing WorkPieces.
    """
    mock_geo_a = MagicMock(spec=Geometry)
    mock_geo_b = MagicMock(spec=Geometry)
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_LAYERS)
    plan = [
        LayoutItem(
            layer_id="LayerA",
            layer_name="LayerA",
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(0, 0, 10, 10),
        ),
        LayoutItem(
            layer_id="LayerB",
            layer_name="LayerB",
            world_matrix=Matrix.translation(10, 10),
            normalization_matrix=Matrix.identity(),
            crop_window=(20, 20, 10, 10),
        ),
    ]
    geometries: dict[str | None, Geometry] = {
        "LayerA": mock_geo_a,
        "LayerB": mock_geo_b,
    }

    items = assembler.create_items(source, plan, spec, "TestFile", geometries)

    assert len(items) == 2

    layer_a = items[0]
    assert isinstance(layer_a, Layer)
    if isinstance(layer_a, Layer):
        assert layer_a.name == "LayerA"
        assert len(layer_a.children) == 2
        wp_a = next(c for c in layer_a.children if isinstance(c, WorkPiece))
        if wp_a.source_segment:
            assert wp_a.source_segment.layer_id == "LayerA"
            assert wp_a.source_segment.pristine_geometry is mock_geo_a

    layer_b = items[1]
    assert isinstance(layer_b, Layer)
    if isinstance(layer_b, Layer):
        assert layer_b.name == "LayerB"

        wp_b = next(c for c in layer_b.children if isinstance(c, WorkPiece))
        if wp_b.source_segment:
            assert wp_b.source_segment.pristine_geometry is mock_geo_b
        tx, ty = wp_b.matrix.get_translation()
        assert tx == 10.0
        assert ty == 10.0


def test_grouped_items_share_one_layer(assembler, source):
    """Pieces of the same layer become one Layer, with groups inside."""
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.NEW_LAYERS)
    loose = Geometry()
    loose.move_to(0, 0)
    loose.line_to(5, 5)
    piece_a = Geometry()
    piece_a.move_to(10, 10)
    piece_a.line_to(20, 20)
    piece_b = Geometry()
    piece_b.move_to(60, 60)
    piece_b.line_to(70, 70)
    plan = [
        LayoutItem(
            layer_id="A",
            layer_name="A",
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(0, 0, 20, 20),
            geometry=loose,
        ),
        LayoutItem(
            layer_id="A",
            layer_name="A",
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(10, 10, 10, 10),
            group_id="block-0",
            group_name="Block",
            geometry=piece_a,
        ),
        LayoutItem(
            layer_id="B",
            layer_name="B",
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(60, 60, 10, 10),
            group_id="block-0",
            group_name="Block",
            geometry=piece_b,
        ),
    ]

    items = assembler.create_items(source, plan, spec, "TestFile", {})

    assert len(items) == 2
    layer_a, layer_b = items
    assert isinstance(layer_a, Layer)
    assert isinstance(layer_b, Layer)

    workpieces_a = [
        child for child in layer_a.children if isinstance(child, WorkPiece)
    ]
    assert len(workpieces_a) == 1
    assert workpieces_a[0].source_segment is not None
    assert workpieces_a[0].source_segment.pristine_geometry is loose

    groups_a = [
        child for child in layer_a.children if isinstance(child, Group)
    ]
    assert len(groups_a) == 1
    assert groups_a[0].name == "Block"
    group_wp = groups_a[0].children[0]
    assert isinstance(group_wp, WorkPiece)
    assert group_wp.source_segment is not None
    assert group_wp.source_segment.pristine_geometry is piece_a

    groups_b = [
        child for child in layer_b.children if isinstance(child, Group)
    ]
    assert len(groups_b) == 1
    group_wp_b = groups_b[0].children[0]
    assert isinstance(group_wp_b, WorkPiece)
    assert group_wp_b.source_segment is not None
    assert group_wp_b.source_segment.pristine_geometry is piece_b


def test_flattened_groups_stay_at_the_top_level(assembler, source):
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
    block = Geometry()
    block.move_to(10, 10)
    block.line_to(20, 20)
    loose = Geometry()
    loose.move_to(0, 0)
    loose.line_to(5, 5)
    plan = [
        LayoutItem(
            layer_id=None,
            layer_name=None,
            world_matrix=Matrix.translation(10, 10) @ Matrix.scale(10, 10),
            normalization_matrix=Matrix.identity(),
            crop_window=(10, 10, 10, 10),
            group_id="block-0",
            group_name="Block",
            geometry=block,
        ),
        LayoutItem(
            layer_id=None,
            layer_name=None,
            world_matrix=Matrix.identity(),
            normalization_matrix=Matrix.identity(),
            crop_window=(0, 0, 5, 5),
            geometry=loose,
        ),
    ]

    items = assembler.create_items(source, plan, spec, "TestFile", {})

    assert len(items) == 2
    assert isinstance(items[0], Group)
    assert items[0].name == "Block"
    assert items[0].matrix.get_translation() == pytest.approx((10.0, 10.0))
    assert items[0].matrix.get_abs_scale() == pytest.approx((10.0, 10.0))
    assert isinstance(items[1], WorkPiece)
