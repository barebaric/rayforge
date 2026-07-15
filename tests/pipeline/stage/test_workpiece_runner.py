import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from raygeo.geo import Geometry, Matrix

from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment


from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER


@pytest.fixture
def mock_proxy(adopting_mock_proxy):
    """Mocks the ExecutionContextProxy passed to the subprocess."""
    adopting_mock_proxy.sub_context.return_value = adopting_mock_proxy
    adopting_mock_proxy.parent_log_level = logging.DEBUG
    return adopting_mock_proxy


@pytest.fixture
def base_workpiece():
    """Creates a WorkPiece with basic vector data."""
    # The geometry created here represents the Y-down, normalized mask.
    geo = Geometry()
    geo.move_to(0, 0, 0)
    geo.line_to(1, 0, 0)
    geo.line_to(1, 1, 0)
    geo.line_to(0, 1, 0)
    geo.close_path()

    # In a real app, a SourceAsset would be created by the importer.
    # We simulate a minimal one here.
    source = SourceAsset(
        source_file=Path("test.dxf"), original_data=b"", renderer=MagicMock()
    )

    # Create the segment that defines the shape.
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    # Create the workpiece using the new constructor.
    wp = WorkPiece(name="test_wp", source_segment=segment)
    wp.set_size(25, 25)  # Set a physical size
    return wp


# This data is used by multiple tests to create the SourceAsset.
SVG_DATA = b"""
<svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
<rect width="50" height="30" fill="black"/>
</svg>"""


@pytest.fixture
def rasterable_workpiece():
    """
    Creates a WorkPiece with a renderer and data, suitable for raster ops.
    """
    # Create a dummy 1x1 geometry for the segment mask.
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 0)
    geo.line_to(1, 1)
    geo.line_to(0, 1)
    geo.close_path()

    source = SourceAsset(
        source_file=Path("raster.svg"),
        original_data=SVG_DATA,
        renderer=SVG_RENDERER,
    )
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    wp = WorkPiece(name="raster_wp.svg", source_segment=segment)
    # The _data and _renderer are hydrated for the subprocess test.
    wp._data = SVG_DATA
    wp._renderer = SVG_RENDERER
    wp.set_size(50, 30)
    return wp
