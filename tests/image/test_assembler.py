from typing import Dict
from unittest.mock import MagicMock
import pytest
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image.assembler import ItemAssembler
from rayforge.image.structures import LayoutItem
from rayforge.core.source_asset import SourceAsset
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece


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
    geometries: Dict[str | None, Geometry] = {"some_layer": mock_geo}

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
    spec = PassthroughSpec(create_new_layers=True)
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
    geometries: Dict[str | None, Geometry] = {
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
