import pytest
from raygeo.geo import Geometry, Matrix

from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.path_cleanup_cmd import PathCleanupCmd
from rayforge.doceditor.split_cmd import ContourSplitStrategy

WIDTH_MM = 100.0
HEIGHT_MM = 50.0


@pytest.fixture
def cleanup_cmd(doc_editor):
    return PathCleanupCmd(doc_editor)


def _mm_polyline(points_mm, close=False) -> Geometry:
    """A polyline given in mm, normalized to the 100x50 mm test box."""
    geo = Geometry()
    pts = [(x / WIDTH_MM, y / HEIGHT_MM) for x, y in points_mm]
    geo.move_to(*pts[0])
    for p in pts[1:]:
        geo.line_to(*p)
    if close:
        geo.line_to(*pts[0])
    return geo


def _concat(*geos: Geometry) -> Geometry:
    result = Geometry()
    for geo in geos:
        result.extend(geo)
    return result


def _add_workpiece(doc_editor, norm_geo: Geometry, name="Logo", pos=(0, 0)):
    segment = SourceAssetSegment(
        source_asset_uid="dummy",
        pristine_geometry=norm_geo,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )
    wp = WorkPiece(name=name, source_segment=segment)
    wp.natural_width_mm = WIDTH_MM
    wp.natural_height_mm = HEIGHT_MM
    wp.set_size(WIDTH_MM, HEIGHT_MM)
    wp.pos = pos
    doc_editor.doc.active_layer.add_child(wp)
    return wp


def _contours(wp: WorkPiece) -> list[Geometry]:
    assert wp.boundaries is not None
    return wp.boundaries.split_into_contours()


# A 0.3 mm wide shape inset into the 100x50 mm box. The bounding box
# corners at (0, 0) and (100, 50) keep the normalization stable.
FRAME = [(0, 0), (100, 0), (100, 50), (0, 50)]
OPEN_SQUARE_MM = [(20, 10), (40, 10), (40, 30), (20, 30), (20, 10.05)]


class TestClosePaths:
    def test_tolerance_is_in_millimetres(self, doc_editor, cleanup_cmd):
        geo = _concat(
            _mm_polyline(FRAME, close=True), _mm_polyline(OPEN_SQUARE_MM)
        )
        wp = _add_workpiece(doc_editor, geo)

        assert cleanup_cmd.close_paths([wp], tolerance_mm=0.01) == 0
        assert not _contours(wp)[1].is_closed()

        assert cleanup_cmd.close_paths([wp], tolerance_mm=0.1) == 1
        assert _contours(wp)[1].is_closed()

    def test_is_one_undo_step(self, doc_editor, cleanup_cmd):
        geo = _mm_polyline(OPEN_SQUARE_MM)
        wp = _add_workpiece(doc_editor, geo)
        history = doc_editor.history_manager
        depth = len(history.undo_stack)

        cleanup_cmd.close_paths([wp], tolerance_mm=0.1)
        assert len(history.undo_stack) == depth + 1
        assert _contours(wp)[0].is_closed()

        history.undo()
        assert not _contours(wp)[0].is_closed()

    def test_no_change_adds_no_undo_step(self, doc_editor, cleanup_cmd):
        wp = _add_workpiece(doc_editor, _mm_polyline(FRAME, close=True))
        history = doc_editor.history_manager
        depth = len(history.undo_stack)

        assert cleanup_cmd.close_paths([wp], tolerance_mm=0.1) == 0
        assert len(history.undo_stack) == depth

    def test_keeps_size_and_position(self, doc_editor, cleanup_cmd):
        wp = _add_workpiece(doc_editor, _mm_polyline(OPEN_SQUARE_MM))
        size, pos = wp.size, wp.pos
        cleanup_cmd.close_paths([wp], tolerance_mm=0.1)
        assert wp.size == pytest.approx(size)
        assert wp.pos == pytest.approx(pos)

    def test_skips_sketch_based_workpieces(self, doc_editor, cleanup_cmd):
        wp = _add_workpiece(doc_editor, _mm_polyline(OPEN_SQUARE_MM))
        wp.geometry_provider_uid = "some-sketch"
        assert cleanup_cmd.close_paths([wp], tolerance_mm=0.1) == 0
        assert wp._edited_boundaries is None


class TestJoinPaths:
    def test_joins_pieces_within_tolerance(self, doc_editor, cleanup_cmd):
        geo = _concat(
            _mm_polyline(FRAME, close=True),
            _mm_polyline([(20, 10), (40, 10), (40, 30)]),
            _mm_polyline([(20, 10), (20, 30), (40, 30.05)]),
        )
        wp = _add_workpiece(doc_editor, geo)

        assert cleanup_cmd.join_paths([wp], tolerance_mm=0.1) == 1
        contours = _contours(wp)
        assert len(contours) == 2
        assert contours[1].is_closed()

        doc_editor.history_manager.undo()
        assert len(_contours(wp)) == 3


class TestDeleteDuplicates:
    def test_removes_duplicate_contour(self, doc_editor, cleanup_cmd):
        geo = _concat(
            _mm_polyline(FRAME, close=True),
            _mm_polyline(OPEN_SQUARE_MM),
            _mm_polyline(list(reversed(OPEN_SQUARE_MM))),
        )
        wp = _add_workpiece(doc_editor, geo)

        assert cleanup_cmd.delete_duplicates([wp]) == 1
        assert len(_contours(wp)) == 2

        doc_editor.history_manager.undo()
        assert len(_contours(wp)) == 3

    def test_removes_stacked_copy_of_workpiece(self, doc_editor, cleanup_cmd):
        layer = doc_editor.doc.active_layer
        geo = _mm_polyline(FRAME, close=True)
        first = _add_workpiece(doc_editor, geo, name="A", pos=(10, 10))
        copy = _add_workpiece(doc_editor, geo.copy(), name="B", pos=(10, 10))
        moved = _add_workpiece(doc_editor, geo.copy(), name="C", pos=(50, 10))

        assert cleanup_cmd.delete_duplicates([first, copy, moved]) == 1
        assert first in layer.children
        assert copy not in layer.children
        assert moved in layer.children

        doc_editor.history_manager.undo()
        assert copy in layer.children

    def test_nothing_to_delete(self, doc_editor, cleanup_cmd):
        wp = _add_workpiece(doc_editor, _mm_polyline(FRAME, close=True))
        depth = len(doc_editor.history_manager.undo_stack)
        assert cleanup_cmd.delete_duplicates([wp]) == 0
        assert len(doc_editor.history_manager.undo_stack) == depth


class TestBreakApart:
    def test_every_contour_becomes_a_workpiece(self, doc_editor, cleanup_cmd):
        layer = doc_editor.doc.active_layer
        inner = [(10, 10), (90, 10), (90, 40), (10, 40)]
        geo = _concat(
            _mm_polyline(FRAME, close=True),
            _mm_polyline(inner, close=True),
            _mm_polyline([(20, 20), (80, 20)]),
        )
        wp = _add_workpiece(doc_editor, geo)

        new_items = cleanup_cmd.break_apart([wp])
        assert len(new_items) == 3
        assert wp not in layer.children
        sizes = sorted(round(item.size[0]) for item in new_items)
        assert sizes == [60, 80, 100]

        doc_editor.history_manager.undo()
        assert wp in layer.children

    def test_contour_strategy_keeps_open_paths(self, doc_editor):
        geo = _concat(
            _mm_polyline(FRAME, close=True),
            _mm_polyline([(20, 20), (80, 20)]),
        )
        wp = _add_workpiece(doc_editor, geo)
        fragments = ContourSplitStrategy().calculate_fragments(wp)
        assert len(fragments) == 2
