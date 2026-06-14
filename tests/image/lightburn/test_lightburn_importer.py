"""Tests for the LightBurn (.lbrn / .lbrn2) importer."""

import math
from pathlib import Path
from typing import Dict, List

from rayforge.image.structures import (
    ImportManifest,
    ParsingResult,
    VectorizationResult,
)
from rayforge.image.lightburn.importer import (
    LightBurnImporter,
    _parse_xform,
    _parse_verts,
    _parse_prims,
    _build_rect,
    _build_ellipse,
    _build_path_from_verts_and_prims,
    _apply_xform_to_geo,
)
from rayforge.core.matrix import Matrix

ASSETS = Path(__file__).parent / "assets"


def _load(name: str) -> bytes:
    return (ASSETS / name).read_bytes()


# --- Unit tests for parser helpers ---


class TestParseXForm:
    def test_identity(self):
        m = _parse_xform("1 0 0 1 0 0")
        assert m.is_identity()

    def test_translation(self):
        m = _parse_xform("1 0 0 1 100 50")
        assert not m.is_identity()
        tx, ty = m.get_translation()
        assert abs(tx - 100.0) < 1e-6
        assert abs(ty - 50.0) < 1e-6

    def test_scale(self):
        m = _parse_xform("2 0 0 3 0 0")
        sx, sy = m.get_scale()
        assert abs(abs(sx) - 2.0) < 1e-6
        assert abs(abs(sy) - 3.0) < 1e-6

    def test_rotation(self):
        angle = 45.0
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        m = _parse_xform(f"{c} {s} {-s} {c} 0 0")
        rot = m.get_rotation()
        assert abs(rot - angle) < 1.0

    def test_invalid_returns_identity(self):
        m = _parse_xform("1 2 3")
        assert m.is_identity()

    def test_empty_returns_identity(self):
        m = _parse_xform("")
        assert m.is_identity()


class TestParseVerts:
    def test_single_vertex(self):
        verts = _parse_verts("V 10 20")
        assert len(verts) == 1
        assert abs(verts[0]["x"] - 10.0) < 1e-6
        assert abs(verts[0]["y"] - 20.0) < 1e-6

    def test_multiple_vertices(self):
        verts = _parse_verts("V 0 0 V 10 10 V 20 0")
        assert len(verts) == 3
        assert abs(verts[2]["x"] - 20.0) < 1e-6

    def test_negative_coordinates(self):
        verts = _parse_verts("V -10 -20 V 0 0")
        assert len(verts) == 2
        assert abs(verts[0]["x"] - (-10.0)) < 1e-6

    def test_with_control_points(self):
        verts = _parse_verts("V 0 0 c0x10c0y20c1x30c1y40 V 50 60")
        assert len(verts) == 2
        v0 = verts[0]
        assert v0.get("c0x") == 10.0
        assert v0.get("c0y") == 20.0
        assert v0.get("c1x") == 30.0
        assert v0.get("c1y") == 40.0

    def test_empty(self):
        verts = _parse_verts("")
        assert len(verts) == 0

    def test_scientific_notation(self):
        verts = _parse_verts("V 1e-5 2e+3")
        assert len(verts) == 1
        assert abs(verts[0]["x"] - 1e-5) < 1e-9
        assert abs(verts[0]["y"] - 2000.0) < 1e-9


class TestParsePrims:
    def test_line_prims(self):
        prims = _parse_prims("L 0 1 L 1 2")
        assert len(prims) == 2
        assert prims[0] == ("L", 0, 1)
        assert prims[1] == ("L", 1, 2)

    def test_bezier_prims(self):
        prims = _parse_prims("B 0 1 B 2 3")
        assert len(prims) == 2
        assert prims[0] == ("B", 0, 1)
        assert prims[1] == ("B", 2, 3)

    def test_mixed_prims(self):
        prims = _parse_prims("L 0 1 B 1 2 L 2 3")
        assert len(prims) == 3
        assert prims[0] == ("L", 0, 1)
        assert prims[1] == ("B", 1, 2)
        assert prims[2] == ("L", 2, 3)

    def test_empty(self):
        prims = _parse_prims("")
        assert len(prims) == 0


# --- Unit tests for geometry builders ---


class TestBuildRect:
    def test_simple_rect(self):
        geo = _build_rect(100, 50, 0)
        assert not geo.is_empty()
        min_x, min_y, max_x, max_y = geo.rect()
        assert abs(min_x - (-50.0)) < 1e-6
        assert abs(min_y - (-25.0)) < 1e-6
        assert abs(max_x - 50.0) < 1e-6
        assert abs(max_y - 25.0) < 1e-6

    def test_rounded_rect(self):
        geo = _build_rect(100, 50, 5)
        assert not geo.is_empty()

    def test_zero_size(self):
        geo = _build_rect(0, 0, 0)
        assert geo.is_empty()


class TestBuildEllipse:
    def test_circle(self):
        geo = _build_ellipse(50, 50)
        assert not geo.is_empty()
        min_x, min_y, max_x, max_y = geo.rect()
        assert abs(min_x - (-50.0)) < 1.0
        assert abs(min_y - (-50.0)) < 1.0

    def test_ellipse(self):
        geo = _build_ellipse(100, 50)
        assert not geo.is_empty()
        min_x, min_y, max_x, max_y = geo.rect()
        assert abs(min_x - (-100.0)) < 1.0
        assert abs(min_y - (-50.0)) < 1.0

    def test_zero_radius(self):
        assert _build_ellipse(0, 0).is_empty()
        assert _build_ellipse(-1, 5).is_empty()


class TestBuildPath:
    def test_line_closed(self):
        verts: List[Dict[str, float]] = [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 10.0},
        ]
        prims = [("L", 0, 1), ("L", 1, 2), ("L", 2, 0)]
        geo = _build_path_from_verts_and_prims(
            verts, prims, "L 0 1 L 1 2 L 2 0"
        )
        assert not geo.is_empty()

    def test_lineclosed_string(self):
        verts: List[Dict[str, float]] = [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 10.0},
            {"x": 0.0, "y": 10.0},
        ]
        geo = _build_path_from_verts_and_prims(verts, [], "LineClosed")
        assert not geo.is_empty()

    def test_empty_verts(self):
        geo = _build_path_from_verts_and_prims([], [], "")
        assert geo.is_empty()

    def test_single_vert_lineclosed(self):
        verts: List[Dict[str, float]] = [{"x": 5.0, "y": 5.0}]
        geo = _build_path_from_verts_and_prims(verts, [], "LineClosed")
        assert not geo.is_empty()

    def test_bezier(self):
        verts: List[Dict[str, float]] = [
            {"x": 0.0, "y": 0.0, "c0x": 10.0, "c0y": 0.0,
             "c1x": 10.0, "c1y": 0.0},
            {"x": 20.0, "y": 0.0, "c1x": 20.0, "c1y": 10.0,
             "c0x": 20.0, "c0y": 10.0},
        ]
        prims = [("B", 0, 1)]
        geo = _build_path_from_verts_and_prims(verts, prims, "B 0 1")
        assert not geo.is_empty()

    def test_bezier_missing_controls_falls_back_to_line(self):
        verts: List[Dict[str, float]] = [
            {"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0}
        ]
        prims = [("B", 0, 1)]
        geo = _build_path_from_verts_and_prims(verts, prims, "B 0 1")
        assert not geo.is_empty()


class TestApplyXFormToGeo:
    def test_translation(self):
        geo = _build_rect(10, 10, 0)
        xform = _parse_xform("1 0 0 1 100 50")
        geo = _apply_xform_to_geo(geo, xform)
        min_x, min_y, max_x, max_y = geo.rect()
        assert abs(min_x - 95.0) < 1e-6
        assert abs(min_y - 45.0) < 1e-6

    def test_identity(self):
        geo = _build_rect(10, 10, 0)
        orig = geo.copy()
        xform = Matrix()
        geo = _apply_xform_to_geo(geo, xform)
        rects_match = geo.rect() == orig.rect()
        assert rects_match


# --- Integration tests for the importer ---


class TestLightBurnImporter:
    def _import(self, filename: str) -> LightBurnImporter:
        data = _load(filename)
        return LightBurnImporter(data, source_file=ASSETS / filename)

    def test_import_lbrn2_with_rect(self):
        importer = self._import("rect.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_import_lbrn_with_rect(self):
        importer = self._import("rect_legacy.lbrn")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_import_with_ellipse(self):
        importer = self._import("ellipse.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_import_with_path(self):
        importer = self._import("path.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_scan_returns_manifest(self):
        importer = self._import("rect.lbrn2")
        manifest = importer.scan()
        assert isinstance(manifest, ImportManifest)
        assert manifest.title is not None

    def test_parse_returns_parsing_result(self):
        importer = self._import("rect.lbrn2")
        result = importer.parse()
        assert result is not None
        assert isinstance(result, ParsingResult)
        assert result.native_unit_to_mm == 1.0
        assert result.is_y_down is False

    def test_vectorize_returns_vectorization_result(self):
        importer = self._import("rect.lbrn2")
        parse_result = importer.parse()
        assert parse_result is not None
        from rayforge.core.vectorization_spec import PassthroughSpec
        vec_result = importer.vectorize(parse_result, PassthroughSpec())
        assert isinstance(vec_result, VectorizationResult)
        assert len(vec_result.geometries_by_layer) > 0

    def test_create_source_asset(self):
        importer = self._import("rect.lbrn2")
        parse_result = importer.parse()
        assert parse_result is not None
        asset = importer.create_source_asset(parse_result)
        assert asset is not None
        assert asset.width_mm > 0
        assert asset.height_mm > 0

    def test_invalid_xml_returns_errors(self):
        importer = LightBurnImporter(
            b"not xml", source_file=ASSETS / "bad.lbrn2"
        )
        manifest = importer.scan()
        assert len(manifest.errors) > 0

    def test_empty_file_handled_gracefully(self):
        importer = self._import("empty.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) == 0

    def test_multiple_layers_detected(self):
        importer = self._import("multi_layer.lbrn2")
        manifest = importer.scan()
        assert len(manifest.layers) == 2
        parse_result = importer.parse()
        assert parse_result is not None
        assert len(parse_result.layers) == 2

    def test_group_shape(self):
        importer = self._import("group.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_extension_registration(self):
        from rayforge.image.registry import importer_registry
        importer_cls = importer_registry.get_by_extension(".lbrn2")
        assert importer_cls is not None
        assert importer_cls is LightBurnImporter
        importer_cls = importer_registry.get_by_extension(".lbrn")
        assert importer_cls is not None
        assert importer_cls is LightBurnImporter

    def test_real_grid(self):
        self._run("grid.lbrn2")

    def test_real_printable_area_jig(self):
        self._run("printable_area_jig.lbrn2")

    def test_real_switch_plate(self):
        self._run("switch_plate.lbrn2")

    def test_real_fence(self):
        self._run("fence.lbrn2")

    def test_real_board_with_soldermask(self):
        self._run("board-with-soldermask-only.lbrn2")

    def test_real_images(self):
        importer = self._import("images.lbrn2")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None

    def _run(self, filename: str):
        importer = self._import(filename)
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0

    def test_roundtrip_bounds(self):
        importer = self._import("rect_legacy.lbrn")
        result = importer.get_doc_items()
        assert result is not None
        assert result.payload is not None
        assert len(result.payload.items) > 0
