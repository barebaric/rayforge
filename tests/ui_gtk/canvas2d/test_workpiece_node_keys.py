from unittest.mock import MagicMock

import pytest
from gi.repository import Gdk
from raygeo.geo import Geometry, Matrix

from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.canvas2d.elements.workpiece import (
    VectorEditState,
    WorkPieceElement,
)


def _square() -> Geometry:
    geo = Geometry()
    geo.move_to(0.0, 0.0)
    for x, y in ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)):
        geo.line_to(x, y)
    return geo


@pytest.fixture
def element(doc_editor):
    segment = SourceAssetSegment(
        source_asset_uid="dummy",
        pristine_geometry=_square(),
        normalization_matrix=Matrix.identity(),
        vectorization_spec=PassthroughSpec(),
    )
    wp = WorkPiece(name="Logo", source_segment=segment)
    wp.natural_width_mm = 40.0
    wp.natural_height_mm = 40.0
    wp.set_size(40.0, 40.0)
    doc_editor.doc.active_layer.add_child(wp)
    elem = WorkPieceElement.__new__(WorkPieceElement)
    elem.data = wp
    elem.canvas = MagicMock()
    elem.canvas.editor = doc_editor
    boundaries = wp.boundaries
    assert boundaries is not None
    elem._edit_state = VectorEditState(boundaries)
    return elem


@pytest.mark.ui
@pytest.mark.asyncio
async def test_insert_clears_the_shifted_segment_selection(element):
    state = element._edit_state
    state.selected_segments = {3}
    state.hovered_segment = 1
    state.hovered_t = 0.5
    assert element.handle_edit_key(Gdk.KEY_i) is True
    state = element._edit_state
    assert len(state.geometry.data) == 6
    assert state.selected_segments == set()
    assert state.selected_node is not None


@pytest.mark.ui
@pytest.mark.asyncio
async def test_delete_removes_the_selected_node_first(element):
    element._edit_state.selected_segments = {3}
    element._edit_state.selected_node = 2
    assert element.handle_edit_key(Gdk.KEY_Delete) is True
    assert len(element._edit_state.geometry.data) == 4
    assert element._edit_state.selected_segments == set()


@pytest.mark.ui
@pytest.mark.asyncio
async def test_delete_segments_without_a_selected_node(element):
    element._edit_state.selected_segments = {2}
    assert element.handle_edit_key(Gdk.KEY_Delete) is True
    assert len(element._edit_state.geometry.data) < 5
    assert element._edit_state.selected_segments == set()


@pytest.mark.ui
@pytest.mark.asyncio
async def test_other_keys_are_not_handled(element):
    assert element.handle_edit_key(Gdk.KEY_z) is False
