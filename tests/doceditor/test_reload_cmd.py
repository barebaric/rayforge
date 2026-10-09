import math
from pathlib import Path
from unittest.mock import MagicMock

import ezdxf
import pytest
import pyvips
from raygeo.geo import Geometry, Matrix

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.source_asset import SourceAsset
from rayforge.core.tab import Tab
from rayforge.core.vectorization_spec import (
    LayerImportMode,
    LayerSource,
    PassthroughSpec,
    TraceSpec,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.reload_cmd import (
    MAX_LISTED_TABS,
    TAB_TOLERANCE_MM,
    DroppedTab,
    ReloadReport,
)
from rayforge.image import importer_registry
from rayforge.shared.tasker.manager import TaskManager

PAGE_80 = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="80mm" '
    'viewBox="0 0 100 80">{body}</svg>'
)
PAGE_120 = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="120mm" '
    'viewBox="0 0 100 120">{body}</svg>'
)
RED_RECT = (
    '<rect x="10" y="10" width="20" height="10" fill="none" stroke="#ff0000"/>'
)
BLUE_CIRCLE = '<circle cx="60" cy="40" r="10" fill="none" stroke="#0000ff"/>'
GREEN_SQUARE = (
    '<rect x="2" y="60" width="5" height="5" fill="none" stroke="#00ff00"/>'
)
WIDE_RED_RECT = (
    '<rect x="10" y="10" width="30" height="10" fill="none" stroke="#ff0000"/>'
)


@pytest.fixture
def editor(context_initializer):
    task_manager = MagicMock(spec=TaskManager)
    editor = DocEditor(task_manager, context_initializer, Doc())
    yield editor
    editor.cleanup()


def _write(path: Path, template: str, *parts: str) -> Path:
    path.write_text(template.format(body="".join(parts)))
    return path


def _import(editor: DocEditor, path: Path, spec=None) -> SourceAsset:
    importer_cls = importer_registry.get_for_file(path)
    assert importer_cls is not None
    importer = importer_cls(path.read_bytes(), path)
    result = importer.get_doc_items(spec)
    assert result is not None and result.payload is not None
    editor.file._finalize_import_on_main_thread(
        result.payload, path, None, spec
    )
    assert result.payload.source is not None
    return result.payload.source


def _flatten_spec() -> PassthroughSpec:
    return PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)


def _color_spec(active=None) -> PassthroughSpec:
    return PassthroughSpec(
        active_layer_ids=active,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
        layer_source=LayerSource.COLORS,
    )


def _workpieces_of(doc: Doc, asset: SourceAsset) -> list[WorkPiece]:
    return [
        wp
        for wp in doc.all_workpieces
        if wp.source_segment
        and wp.source_segment.source_asset_uid == asset.uid
    ]


def _by_layer_id(doc: Doc, asset: SourceAsset) -> dict:
    return {
        wp.source_segment.layer_id: wp
        for wp in _workpieces_of(doc, asset)
        if wp.source_segment
    }


def _sample_points(geo: Geometry) -> list[tuple[float, float]]:
    points = []
    for index in range(len(list(geo.iter_commands()))):
        for t in (0.0, 0.5, 1.0):
            point = geo.get_point_at(index, t)
            if point is not None:
                points.append((point[0], point[1]))
    return points


def _distance_to(geo: Geometry, x: float, y: float) -> float:
    closest = geo.find_closest_point(x, y)
    assert closest is not None
    point = closest[2]
    return math.hypot(point[0] - x, point[1] - y)


def _assert_contained(old: Geometry, new: Geometry, tol: float = 1e-3):
    points = _sample_points(old)
    assert points
    for x, y in points:
        assert _distance_to(new, x, y) < tol, (x, y)


def _world_bbox(wp: WorkPiece) -> tuple[float, float, float, float]:
    geo = wp.get_world_geometry()
    assert geo is not None
    return geo.rect()


def _transform(wp: WorkPiece, delta: Matrix):
    wp.matrix = delta @ wp.matrix


class TestReloadPlacement:
    def test_unchanged_content_stays_put_after_user_transform(
        self, editor, tmp_path
    ):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(
            wp,
            Matrix.translation(13, -4)
            @ Matrix.rotation(30)
            @ Matrix.scale(1.5, 1.5),
        )
        old_world = wp.get_world_geometry()
        assert old_world is not None

        _write(path, PAGE_80, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.updated == 1
        new_world = wp.get_world_geometry()
        assert new_world is not None
        _assert_contained(old_world, new_world)
        assert len(_sample_points(new_world)) > len(_sample_points(old_world))

    def test_taller_page_does_not_move_content(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        old_world = wp.get_world_geometry()
        assert old_world is not None

        _write(path, PAGE_120, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        editor.reload.reload_source(asset)

        new_world = wp.get_world_geometry()
        assert new_world is not None
        _assert_contained(old_world, new_world)

    def test_scale_factor_is_kept(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(wp, Matrix.scale(2.0, 2.0))
        x0, y0, x1, y1 = _world_bbox(wp)
        assert x1 - x0 == pytest.approx(40.0, abs=0.1)

        _write(path, PAGE_80, WIDE_RED_RECT)
        editor.reload.reload_source(asset)

        nx0, ny0, nx1, ny1 = _world_bbox(wp)
        assert nx1 - nx0 == pytest.approx(60.0, abs=0.1)
        assert ny1 - ny0 == pytest.approx(y1 - y0, abs=0.1)
        assert nx0 == pytest.approx(x0, abs=0.1)
        assert ny1 == pytest.approx(y1, abs=0.1)

    def test_copies_keep_their_own_placement(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, path, _flatten_spec())
        (original,) = _workpieces_of(editor.doc, asset)
        copy = WorkPiece.from_dict(original.to_dict())
        copy.uid = "copy-uid"
        assert original.parent is not None
        original.parent.add_child(copy)
        _transform(copy, Matrix.translation(50, 25))
        old_original = original.get_world_geometry()
        old_copy = copy.get_world_geometry()
        assert old_original is not None and old_copy is not None

        _write(path, PAGE_80, RED_RECT, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.updated == 2
        new_original = original.get_world_geometry()
        new_copy = copy.get_world_geometry()
        assert new_original is not None and new_copy is not None
        _assert_contained(old_original, new_original)
        _assert_contained(old_copy, new_copy)


class TestReloadKeepsSettings:
    def test_identity_layer_and_spec_are_kept(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        spec = _flatten_spec()
        asset = _import(editor, path, spec)
        (wp,) = _workpieces_of(editor.doc, asset)
        wp.name = "My part"
        wp.tabs_enabled = False
        assert wp.source_segment is not None
        wp.source_segment.image_modifier_chain = [{"name": "Invert"}]
        uid, parent = wp.uid, wp.parent
        old_width = wp.natural_width_mm
        old_spec = wp.source_segment.vectorization_spec

        _write(path, PAGE_80, WIDE_RED_RECT)
        editor.reload.reload_source(asset)

        assert wp.uid == uid
        assert wp.parent is parent
        assert wp.name == "My part"
        assert wp.tabs_enabled is False
        assert wp.source_segment is not None
        assert wp.source_segment.source_asset_uid == asset.uid
        assert wp.source_segment.image_modifier_chain == [{"name": "Invert"}]
        assert (
            wp.source_segment.vectorization_spec.to_dict()
            == old_spec.to_dict()
        )
        assert wp.natural_width_mm > old_width + 9

    def test_source_asset_takes_new_data(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, path, _flatten_spec())
        uid = asset.uid

        _write(path, PAGE_80, WIDE_RED_RECT)
        editor.reload.reload_source(asset)

        assert asset.uid == uid
        assert asset.original_data == path.read_bytes()
        assert editor.doc.get_source_asset_by_uid(uid) is asset

    def test_tab_on_unchanged_outline_is_kept(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(wp, Matrix.translation(5, 5))
        assert wp.boundaries is not None
        closest = wp.boundaries.find_closest_point(0.0, 1.0)
        assert closest is not None
        wp.tabs = [Tab(width=3.0, segment_index=closest[0], pos=closest[1])]
        tab_point = _tab_world_point(wp, wp.tabs[0])

        _write(path, PAGE_80, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.tabs_dropped == 0
        assert len(wp.tabs) == 1
        assert wp.tabs[0].width == 3.0
        new_point = _tab_world_point(wp, wp.tabs[0])
        assert math.dist(tab_point, new_point) < 1e-3

    def test_tab_on_moved_outline_is_dropped(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        assert wp.boundaries is not None
        closest = wp.boundaries.find_closest_point(0.0, 1.0)
        assert closest is not None
        wp.tabs = [Tab(width=3.0, segment_index=closest[0], pos=closest[1])]

        moved = RED_RECT.replace('x="10"', 'x="70"').replace(
            'y="10"', 'y="60"'
        )
        _write(path, PAGE_80, moved, BLUE_CIRCLE)
        report = editor.reload.reload_source(asset)

        assert report.tabs_dropped == 1
        assert wp.tabs == []

    def test_dropped_tab_reports_where_it_was(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        wp.name = "Lid"
        _transform(wp, Matrix.translation(5, 7))
        assert wp.boundaries is not None
        closest = wp.boundaries.find_closest_point(0.0, 1.0)
        assert closest is not None
        tab = Tab(width=3.0, segment_index=closest[0], pos=closest[1])
        wp.tabs = [tab]
        x, y = _tab_world_point(wp, tab)

        moved = RED_RECT.replace('x="10"', 'x="70"').replace(
            'y="10"', 'y="60"'
        )
        _write(path, PAGE_80, moved, BLUE_CIRCLE)
        report = editor.reload.reload_source(asset)

        (dropped,) = report.dropped_tabs
        assert dropped.workpiece_name == "Lid"
        assert dropped.x_mm == pytest.approx(x, abs=1e-3)
        assert dropped.y_mm == pytest.approx(y, abs=1e-3)
        assert dropped.distance_mm > TAB_TOLERANCE_MM
        message = report.describe()
        assert f"Lid ({x:.1f}, {y:.1f} mm)" in message
        assert f"{TAB_TOLERANCE_MM:.1f} mm" in message

    def test_kept_tabs_are_not_reported(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        assert wp.boundaries is not None
        closest = wp.boundaries.find_closest_point(0.0, 1.0)
        assert closest is not None
        wp.tabs = [Tab(width=3.0, segment_index=closest[0], pos=closest[1])]

        _write(path, PAGE_80, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.dropped_tabs == []
        assert "tab" not in report.describe()

    def test_split_pieces_are_left_alone(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        boundaries = wp.boundaries
        assert boundaries is not None
        pieces = wp.apply_split(boundaries.split_into_components())
        layer = wp.parent
        assert layer is not None
        layer.remove_child(wp)
        for piece in pieces:
            layer.add_child(piece)
        matrices = [piece.matrix.copy() for piece in pieces]

        _write(path, PAGE_80, WIDE_RED_RECT, BLUE_CIRCLE)
        report = editor.reload.reload_source(asset)

        assert report.updated == 0
        assert report.skipped == len(pieces)
        assert [p.matrix for p in pieces] == matrices


def _tab_world_point(wp: WorkPiece, tab: Tab) -> tuple[float, float]:
    boundaries = wp.boundaries
    assert boundaries is not None
    point = boundaries.get_point_at(tab.segment_index, tab.pos)
    assert point is not None
    return wp.get_world_transform().transform_point((point[0], point[1]))


class TestReloadLayers:
    def test_matches_elements_by_color(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _color_spec())
        by_id = _by_layer_id(editor.doc, asset)
        red = by_id["#ff0000"]
        blue = by_id["#0000ff"]
        _transform(red, Matrix.translation(7, 3))
        _transform(blue, Matrix.translation(-20, 11))
        old_blue = blue.get_world_geometry()
        assert old_blue is not None

        _write(path, PAGE_80, WIDE_RED_RECT, BLUE_CIRCLE)
        report = editor.reload.reload_source(asset)

        assert report.updated == 2
        new_blue = blue.get_world_geometry()
        assert new_blue is not None
        _assert_contained(old_blue, new_blue)
        rx0, _ry0, rx1, _ry1 = _world_bbox(red)
        assert rx1 - rx0 == pytest.approx(30.0, abs=0.1)

    def test_element_gone_from_file_is_removed(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _color_spec())

        _write(path, PAGE_80, RED_RECT)
        report = editor.reload.reload_source(asset)

        assert report.removed == 1
        assert set(_by_layer_id(editor.doc, asset)) == {"#ff0000"}

    def test_new_color_is_added_when_all_were_imported(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _color_spec())
        by_id = _by_layer_id(editor.doc, asset)
        _transform(by_id["#ff0000"], Matrix.translation(7, 3))
        _transform(by_id["#0000ff"], Matrix.translation(7, 3))
        layer_count = len(editor.doc.layers)

        _write(path, PAGE_80, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.added == 1
        green = _by_layer_id(editor.doc, asset)["#00ff00"]
        assert isinstance(green.parent, Layer)
        assert len(editor.doc.layers) == layer_count + 1
        gx0, _gy0, _gx1, gy1 = _world_bbox(green)
        rx0, _ry0, _rx1, ry1 = _world_bbox(by_id["#ff0000"])
        assert gx0 == pytest.approx(rx0 - 8.0, abs=0.1)
        assert gy1 == pytest.approx(ry1 - 50.0, abs=0.1)

    def test_new_color_is_not_added_to_a_chosen_subset(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _color_spec(active=["#ff0000"]))

        _write(path, PAGE_80, RED_RECT, BLUE_CIRCLE, GREEN_SQUARE)
        report = editor.reload.reload_source(asset)

        assert report.added == 0
        assert set(_by_layer_id(editor.doc, asset)) == {"#ff0000"}


class TestReloadUndo:
    def test_reload_is_one_undo_step(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, path, _color_spec())
        old_data = asset.original_data
        by_id = _by_layer_id(editor.doc, asset)
        red = by_id["#ff0000"]
        old_matrix = red.matrix.copy()
        old_width = red.natural_width_mm

        _write(path, PAGE_80, WIDE_RED_RECT, GREEN_SQUARE)
        editor.reload.reload_source(asset)
        assert set(_by_layer_id(editor.doc, asset)) == {
            "#ff0000",
            "#00ff00",
        }

        editor.history_manager.undo()

        assert asset.original_data == old_data
        assert red.matrix == old_matrix
        assert red.natural_width_mm == pytest.approx(old_width)
        assert set(_by_layer_id(editor.doc, asset)) == {
            "#ff0000",
            "#0000ff",
        }

        editor.history_manager.redo()

        assert asset.original_data == path.read_bytes()
        rx0, _ry0, rx1, _ry1 = _world_bbox(red)
        assert rx1 - rx0 == pytest.approx(30.0, abs=0.1)

    def test_unreadable_data_changes_nothing(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        old_data = asset.original_data
        old_matrix = wp.matrix.copy()
        can_undo = editor.history_manager.can_undo()
        undo_depth = len(editor.history_manager.undo_stack)

        report = editor.reload.reload_source(asset, b"this is not an svg")

        assert report.error
        assert asset.original_data == old_data
        assert wp.matrix == old_matrix
        assert editor.history_manager.can_undo() == can_undo
        assert len(editor.history_manager.undo_stack) == undo_depth


def _write_png(
    path: Path, squares: list[tuple[int, int, int]], height: int = 160
) -> Path:
    image = pyvips.Image.black(200, height).invert()
    for x, y, size in squares:
        image = image.draw_rect(0, x, y, size, size, fill=True)
    image.cast("uchar").pngsave(str(path))
    return path


class TestReloadBitmap:
    def test_trace_settings_are_kept_and_content_stays_put(
        self, editor, tmp_path
    ):
        path = _write_png(tmp_path / "a.png", [(20, 20, 40)])
        spec = TraceSpec(threshold=0.42, auto_threshold=False)
        asset = _import(editor, path, spec)
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(wp, Matrix.translation(10, 10) @ Matrix.rotation(15))
        old_world = wp.get_world_geometry()
        assert old_world is not None

        _write_png(path, [(20, 20, 40), (120, 180, 30)], height=240)
        report = editor.reload.reload_source(asset)

        assert report.updated == 1
        assert wp.source_segment is not None
        assert wp.source_segment.vectorization_spec.to_dict() == (
            spec.to_dict()
        )
        new_world = wp.get_world_geometry()
        assert new_world is not None
        _assert_contained(old_world, new_world, tol=0.05)
        assert asset.original_data == path.read_bytes()


def _write_dxf(path: Path, extra_circle: bool) -> Path:
    drawing = ezdxf.new()  # type: ignore
    msp = drawing.modelspace()
    msp.add_lwpolyline([(10, 10), (30, 10), (30, 20), (10, 20)], close=True)
    if extra_circle:
        msp.add_circle((80, 90), radius=5)
    drawing.saveas(path)
    return path


class TestReloadDxf:
    def test_unchanged_content_stays_put(self, editor, tmp_path):
        path = _write_dxf(tmp_path / "a.dxf", extra_circle=False)
        asset = _import(editor, path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(wp, Matrix.translation(4, 9) @ Matrix.rotation(-20))
        old_world = wp.get_world_geometry()
        assert old_world is not None

        _write_dxf(path, extra_circle=True)
        report = editor.reload.reload_source(asset)

        assert report.updated == 1
        new_world = wp.get_world_geometry()
        assert new_world is not None
        _assert_contained(old_world, new_world)


class TestRelink:
    def test_relink_reads_new_path_and_keeps_placement(self, editor, tmp_path):
        old_path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE)
        asset = _import(editor, old_path, _flatten_spec())
        (wp,) = _workpieces_of(editor.doc, asset)
        _transform(wp, Matrix.translation(9, 4) @ Matrix.rotation(10))
        old_world = wp.get_world_geometry()
        assert old_world is not None
        (tmp_path / "moved").mkdir()
        new_path = _write(
            tmp_path / "moved" / "b.svg",
            PAGE_80,
            RED_RECT,
            BLUE_CIRCLE,
            GREEN_SQUARE,
        )
        old_path.unlink()

        report = editor.reload.relink_source(asset, new_path)

        assert report.error is None
        assert report.updated == 1
        assert asset.source_file == new_path
        assert asset.name == "b.svg"
        assert asset.original_data == new_path.read_bytes()
        new_world = wp.get_world_geometry()
        assert new_world is not None
        _assert_contained(old_world, new_world)

    def test_relink_is_undoable(self, editor, tmp_path):
        old_path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, old_path, _flatten_spec())
        old_data = asset.original_data
        new_path = _write(tmp_path / "b.svg", PAGE_80, WIDE_RED_RECT)

        editor.reload.relink_source(asset, new_path)
        editor.history_manager.undo()

        assert asset.source_file == old_path
        assert asset.name == "a.svg"
        assert asset.original_data == old_data

    def test_relink_keeps_a_custom_name(self, editor, tmp_path):
        old_path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, old_path, _flatten_spec())
        asset.name = "Logo"
        new_path = _write(tmp_path / "b.svg", PAGE_80, RED_RECT)

        editor.reload.relink_source(asset, new_path)

        assert asset.name == "Logo"

    def test_relink_to_unreadable_path_changes_nothing(self, editor, tmp_path):
        old_path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, old_path, _flatten_spec())
        undo_depth = len(editor.history_manager.undo_stack)

        report = editor.reload.relink_source(asset, tmp_path / "nope.svg")

        assert report.error
        assert asset.source_file == old_path
        assert len(editor.history_manager.undo_stack) == undo_depth


class TestSourceLookup:
    def test_sources_of_selection_are_distinct(self, editor, tmp_path):
        a = _import(
            editor,
            _write(tmp_path / "a.svg", PAGE_80, RED_RECT, BLUE_CIRCLE),
            _color_spec(),
        )
        b = _import(
            editor,
            _write(tmp_path / "b.svg", PAGE_80, RED_RECT),
            _flatten_spec(),
        )
        items = editor.doc.all_workpieces + [WorkPiece(name="plain")]

        sources = editor.reload.sources_of(items)

        assert sorted(s.uid for s in sources) == sorted([a.uid, b.uid])

    def test_reloadable_needs_existing_file(self, editor, tmp_path):
        path = _write(tmp_path / "a.svg", PAGE_80, RED_RECT)
        asset = _import(editor, path, _flatten_spec())
        assert editor.reload.can_reload(asset)
        assert editor.reload.can_relink(asset)

        path.unlink()

        assert not editor.reload.can_reload(asset)
        assert editor.reload.can_relink(asset)


class TestReportText:
    def test_long_tab_lists_are_shortened(self):
        report = ReloadReport(
            source_name="a.svg",
            dropped_tabs=[
                DroppedTab(f"Part {i}", float(i), 2.0, 3.0)
                for i in range(MAX_LISTED_TABS + 2)
            ],
        )

        message = report.describe()

        assert f"{MAX_LISTED_TABS + 2} tabs" in message
        assert f"Part {MAX_LISTED_TABS - 1} (" in message
        assert f"Part {MAX_LISTED_TABS} (" not in message
        assert "and 2 more" in message
