import pytest
from raygeo.geo import Geometry, Matrix

from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.node_edit_cmd import NodeEditCmd

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def node_edit(doc_editor):
    return NodeEditCmd(doc_editor)


def _square() -> Geometry:
    geo = Geometry()
    geo.move_to(*SQUARE[0])
    for point in SQUARE[1:]:
        geo.line_to(*point)
    geo.line_to(*SQUARE[0])
    return geo


def _add(doc_editor, geo: Geometry) -> WorkPiece:
    segment = SourceAssetSegment(
        source_asset_uid="dummy",
        pristine_geometry=geo,
        normalization_matrix=Matrix.identity(),
        vectorization_spec=PassthroughSpec(),
    )
    wp = WorkPiece(name="Logo", source_segment=segment)
    wp.natural_width_mm = 40.0
    wp.natural_height_mm = 40.0
    wp.set_size(40.0, 40.0)
    doc_editor.doc.active_layer.add_child(wp)
    return wp


def _ends(wp: WorkPiece) -> list[tuple[float, float]]:
    boundaries = wp.boundaries
    assert boundaries is not None
    return [(c.end[0], c.end[1]) for c in boundaries.data]


@pytest.mark.asyncio
async def test_editor_exposes_node_edit(doc_editor):
    assert isinstance(doc_editor.node_edit, NodeEditCmd)


@pytest.mark.asyncio
async def test_move_node_is_one_undo_step(doc_editor, node_edit):
    wp = _add(doc_editor, _square())
    index = node_edit.move_node(wp, 2, 1.2, 1.1)
    assert index == 2
    assert _ends(wp)[2] == pytest.approx((1.2, 1.1))
    assert wp.size == pytest.approx((40.0, 40.0))
    doc_editor.history_manager.undo()
    assert wp._edited_boundaries is None
    assert _ends(wp)[2] == (1.0, 1.0)


@pytest.mark.asyncio
async def test_delete_insert_and_toggle(doc_editor, node_edit):
    wp = _add(doc_editor, _square())
    node_edit.delete_node(wp, 1)
    assert len(_ends(wp)) == 4
    index = node_edit.insert_node(wp, 1, 0.5)
    assert len(_ends(wp)) == 5
    node_edit.set_segments_curve(wp, {index}, True)
    boundaries = wp.boundaries
    assert boundaries is not None
    assert type(boundaries.data[index]).__name__ == "Bezier"
    doc_editor.history_manager.undo()
    assert len(_ends(wp)) == 5
    doc_editor.history_manager.undo()
    assert len(_ends(wp)) == 4
    doc_editor.history_manager.undo()
    assert wp._edited_boundaries is None


@pytest.mark.asyncio
async def test_move_handle(doc_editor, node_edit):
    wp = _add(doc_editor, _square())
    node_edit.set_segments_curve(wp, {1}, True)
    node_edit.move_handle(wp, 1, 1, 0.3, 0.4)
    boundaries = wp.boundaries
    assert boundaries is not None
    assert boundaries.data[1].control1[:2] == pytest.approx((0.3, 0.4))


@pytest.mark.asyncio
async def test_removing_everything_is_allowed(doc_editor, node_edit):
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 1)
    wp = _add(doc_editor, geo)
    node_edit.delete_node(wp, 1)
    boundaries = wp.boundaries
    assert boundaries is not None and boundaries.is_empty()


@pytest.mark.asyncio
async def test_sketch_workpieces_are_left_alone(doc_editor, node_edit):
    wp = _add(doc_editor, _square())
    wp.geometry_provider_uid = "some-sketch"
    assert node_edit.move_node(wp, 2, 0.5, 0.5) is None
    assert wp._edited_boundaries is None


@pytest.mark.asyncio
async def test_does_not_merge_into_a_segment_deletion(doc_editor, node_edit):
    wp = _add(doc_editor, _square())
    doc_editor.edit.delete_segments(wp, {4})
    after_delete = _ends(wp)
    node_edit.delete_node(wp, 1)
    doc_editor.history_manager.undo()
    assert _ends(wp) == after_delete


@pytest.mark.asyncio
async def test_two_segment_deletions_stay_two_undo_steps(doc_editor):
    wp = _add(doc_editor, _square())
    doc_editor.edit.delete_segments(wp, {4})
    after_first = _ends(wp)
    doc_editor.edit.delete_segments(wp, {3})
    assert len(_ends(wp)) == 3
    doc_editor.history_manager.undo()
    assert _ends(wp) == after_first
    doc_editor.history_manager.undo()
    assert wp._edited_boundaries is None
