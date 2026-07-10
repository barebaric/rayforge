from __future__ import annotations

import base64
import logging
import math
import re
import warnings
from dataclasses import dataclass
from gettext import gettext as _
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from raygeo.geo import Geometry

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from raygeo.geo import Matrix

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    TraceSpec,
    VectorizationSpec,
)
from ...image import util
from ...image.geo_renderer import render_geometry_to_png
from ...image.tracing import trace_surface
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ..engine import NormalizationEngine
from ..structures import (
    ImportManifest,
    LayerGeometry,
    LayerInfo,
    ParsingResult,
    VectorizationResult,
)
from .renderer import LIGHTBURN_RENDERER

logger = logging.getLogger(__name__)


_XFORM_RE = re.compile(r"\s+")
_VERTLIST_RE = re.compile(
    r"V\s*"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"\s*"
    r"((?:c0[xXyY]-?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)?"
    r"|c1[xXyY]-?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)?)*)"  # noqa: E501
)
_CONTROL_PT_RE = re.compile(
    r"(c0[xXyY]|c1[xXyY])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
_PRIMLIST_ITEM_RE = re.compile(r"([A-Za-z])\s*(-?\d+(?:\s+-?\d+)*)?")


def _parse_xform(text: str) -> Matrix:
    parts = _XFORM_RE.split(text.strip())
    if len(parts) != 6:
        logger.warning("Invalid XForm string: %s, using identity", text)
        return Matrix()
    try:
        a, b, c, d, tx, ty = (float(v) for v in parts)
    except ValueError:
        logger.warning("Invalid XForm values: %s, using identity", text)
        return Matrix()
    return Matrix(
        [
            [a, c, tx],
            [b, d, ty],
            [0, 0, 1],
        ]
    )


def _parse_verts(text: str) -> List[Dict[str, float]]:
    verts: List[Dict[str, float]] = []
    for match in _VERTLIST_RE.finditer(text):
        x = float(match.group(1))
        y = float(match.group(2))
        v: Dict[str, float] = {"x": x, "y": y}
        cp_str = match.group(3)
        for cp_match in _CONTROL_PT_RE.finditer(cp_str):
            key = cp_match.group(1).lower()
            value = float(cp_match.group(2))
            v[key] = value
        verts.append(v)
    return verts


def _parse_prims(text: str) -> List[Tuple[str, int, int]]:
    prims: List[Tuple[str, int, int]] = []
    for match in _PRIMLIST_ITEM_RE.finditer(text):
        prim_type = match.group(1)
        args_str = match.group(2)
        if prim_type == "L" and args_str:
            parts = [int(x) for x in args_str.split()]
            if len(parts) >= 2:
                prims.append(("L", parts[0], parts[1]))
        elif prim_type == "B" and args_str:
            parts = [int(x) for x in args_str.split()]
            if len(parts) >= 2:
                prims.append(("B", parts[0], parts[1]))
    return prims


def _apply_xform_to_geo(geo: Geometry, xform: Matrix) -> Geometry:
    if xform.is_identity():
        return geo
    geo = geo.copy()
    geo.transform(xform)
    return geo


def _build_rect(w: float, h: float, cr: float) -> Geometry:
    geo = Geometry()
    if w <= 0 or h <= 0:
        return geo
    hw, hh = w / 2.0, h / 2.0

    if cr <= 0:
        geo.move_to(-hw, -hh)
        geo.line_to(hw, -hh)
        geo.line_to(hw, hh)
        geo.line_to(-hw, hh)
        geo.close_path()
    else:
        cr = min(cr, hw, hh)
        segments = 8
        pts = _rounded_rect_points(-hw, -hh, w, h, cr, segments)
        geo.move_to(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            geo.line_to(px, py)
        geo.close_path()
    return geo


def _rounded_rect_points(
    x: float, y: float, w: float, h: float, r: float, seg: int
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(seg + 1):
        a = (math.pi / 2) * (i / seg)
        pts.append((x + r - r * math.cos(a), y + r - r * math.sin(a)))
    for i in range(seg + 1):
        a = (math.pi / 2) * (i / seg)
        pts.append((x + w - r + r * math.sin(a), y + r - r * math.cos(a)))
    for i in range(seg + 1):
        a = (math.pi / 2) * (i / seg)
        pts.append((x + w - r + r * math.cos(a), y + h - r + r * math.sin(a)))
    for i in range(seg + 1):
        a = (math.pi / 2) * (i / seg)
        pts.append((x + r - r * math.sin(a), y + h - r + r * math.cos(a)))
    return pts


def _build_ellipse(rx: float, ry: float) -> Geometry:
    geo = Geometry()
    if rx <= 0 or ry <= 0:
        return geo

    n_segments = 32
    pts: List[Tuple[float, float]] = []
    for i in range(n_segments):
        a = 2 * math.pi * i / n_segments
        pts.append((rx * math.cos(a), ry * math.sin(a)))

    if pts:
        geo.move_to(pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            geo.line_to(px, py)
        geo.close_path()

    return geo


def _build_path_from_verts_and_prims(
    verts: List[Dict[str, float]],
    prims: List[Tuple[str, int, int]],
    prim_list_raw: str,
) -> Geometry:
    geo = Geometry()
    if not verts:
        return geo

    if prim_list_raw.strip() == "LineClosed" or not prims:
        if len(verts) == 1:
            geo.move_to(verts[0]["x"], verts[0]["y"])
            geo.close_path()
        elif len(verts) > 1:
            geo.move_to(verts[0]["x"], verts[0]["y"])
            for v in verts[1:]:
                geo.line_to(v["x"], v["y"])
            geo.close_path()
        return geo

    for prim_type, si, ei in prims:
        if si < 0 or si >= len(verts) or ei < 0 or ei >= len(verts):
            logger.warning("Path primitive index out of range: %d, %d", si, ei)
            continue
        sv = verts[si]
        ev = verts[ei]
        sx, sy = sv["x"], sv["y"]
        ex, ey = ev["x"], ev["y"]

        need_move = geo.is_empty()
        if not need_move:
            lx, ly, _ = geo.get_last_point()
            if abs(lx - sx) > 1e-8 or abs(ly - sy) > 1e-8:
                need_move = True

        if need_move:
            geo.move_to(sx, sy)

        if prim_type == "L":
            geo.line_to(ex, ey)
        elif prim_type == "B":
            c0x = sv.get("c0x")
            c0y = sv.get("c0y")
            c1x = ev.get("c1x")
            c1y = ev.get("c1y")
            if (
                c0x is not None
                and c0y is not None
                and c1x is not None
                and c1y is not None
            ):
                geo.bezier_to(ex, ey, c0x, c0y, c1x, c1y)
            else:
                geo.line_to(ex, ey)

    return geo


def _build_path_text(shape_elem: ET.Element) -> Optional[Geometry]:
    backup_path = shape_elem.find("BackupPath")
    if backup_path is None:
        return None
    bp_shape = backup_path.find("Shape")
    if bp_shape is None:
        return None
    if bp_shape.get("Type") != "Path":
        return None
    vert_list_el = bp_shape.find("VertList")
    prim_list_el = bp_shape.find("PrimList")
    if vert_list_el is None or vert_list_el.text is None:
        return None
    verts = _parse_verts(vert_list_el.text)
    prim_list_raw = (
        prim_list_el.text
        if prim_list_el is not None and prim_list_el.text
        else ""
    )
    prims = _parse_prims(prim_list_raw)
    return _build_path_from_verts_and_prims(verts, prims, prim_list_raw)


@dataclass
class BitmapInfo:
    cut_index: int
    xform: Matrix
    width: float
    height: float
    png_data: bytes


def _shape_to_geometry(
    shape_elem: ET.Element,
    cut_settings: Dict[int, Dict[str, Any]],
    bitmaps: Optional[List[BitmapInfo]] = None,
) -> Optional[Tuple[int, Geometry]]:
    shape_type = shape_elem.get("Type")
    cut_index = int(shape_elem.get("CutIndex", "0"))

    xform_el = shape_elem.find("XForm")
    xform = Matrix()
    if xform_el is not None and xform_el.text:
        xform = _parse_xform(xform_el.text)

    geo: Optional[Geometry] = None

    if shape_type == "Rect":
        w = float(shape_elem.get("W", "0"))
        h = float(shape_elem.get("H", "0"))
        cr = float(shape_elem.get("Cr", "0"))
        if w > 0 and h > 0:
            geo = _build_rect(w, h, cr)

    elif shape_type == "Ellipse":
        rx = float(shape_elem.get("Rx", "0"))
        ry = float(shape_elem.get("Ry", "0"))
        if rx > 0 and ry > 0:
            geo = _build_ellipse(rx, ry)

    elif shape_type == "Path":
        vert_list_el = shape_elem.find("VertList")
        prim_list_el = shape_elem.find("PrimList")
        if vert_list_el is not None and vert_list_el.text is not None:
            verts = _parse_verts(vert_list_el.text)
            prim_list_raw = (
                prim_list_el.text
                if prim_list_el is not None and prim_list_el.text
                else ""
            )
            prims = _parse_prims(prim_list_raw)
            geo = _build_path_from_verts_and_prims(verts, prims, prim_list_raw)

    elif shape_type == "Text":
        has_backup = shape_elem.get("HasBackupPath", "0") == "1"
        if has_backup:
            geo = _build_path_text(shape_elem)

    elif shape_type == "Group":
        children_elem = shape_elem.find("Children")
        if children_elem is not None:
            combined = Geometry()
            for child in children_elem.findall("Shape"):
                child_result = _shape_to_geometry(child, cut_settings, bitmaps)
                if child_result is not None:
                    child_cut_idx, child_geo = child_result
                    child_geo = _apply_xform_to_geo(child_geo, xform)
                    combined.extend(child_geo)
            if not combined.is_empty():
                return (cut_index, combined)
        return None

    elif shape_type == "Bitmap":
        w = float(shape_elem.get("W", "0"))
        h = float(shape_elem.get("H", "0"))
        data_b64 = shape_elem.get("Data", "")
        if w > 0 and h > 0 and data_b64:
            try:
                png_bytes = base64.b64decode(data_b64)
            except Exception:
                logger.warning("Failed to decode Bitmap data")
                return None
            if bitmaps is not None:
                bitmaps.append(
                    BitmapInfo(
                        cut_index=cut_index,
                        xform=xform,
                        width=w,
                        height=h,
                        png_data=png_bytes,
                    )
                )
            geo = _build_rect(w, h, cr=0)
        else:
            return None

    if geo is None or geo.is_empty():
        return None

    geo = _apply_xform_to_geo(geo, xform)
    return (cut_index, geo)


def _build_step_config(
    cs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Translate LightBurn cut settings to generic step configuration."""
    config: Dict[str, Any] = {}
    max_power = cs.get("maxPower")
    if max_power is not None:
        config["power"] = float(max_power) / 100.0
    speed = cs.get("speed")
    if speed is not None:
        config["cut_speed"] = round(float(speed) * 60.0)
    kerf = cs.get("kerf")
    if kerf is not None:
        config["kerf_mm"] = float(kerf)
    num_passes = cs.get("numPasses")
    if num_passes is not None:
        config["passes"] = int(num_passes)
    return config or None


class LightBurnImporter(Importer):
    label = "LightBurn project files"
    mime_types = ("application/x-lightburn",)
    extensions = (".lbrn", ".lbrn2")
    features = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
        ImporterFeature.BITMAP_TRACING,
    }

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._geometries_by_layer: Dict[str, Geometry] = {}
        self._cut_settings: Dict[int, Dict[str, Any]] = {}
        self._project_title: str = ""
        self._bitmaps: List[BitmapInfo] = []

    def scan(self) -> ImportManifest:
        try:
            root = ET.fromstring(self.raw_data)
        except ET.ParseError as e:
            logger.warning("LightBurn scan failed: %s", e)
            self.add_error(_(f"LightBurn file is invalid XML: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

        project = root.find("LightBurnProject")
        if project is None:
            project = root

        title = project.get("AppVersion", self.source_file.stem)
        layers: List[LayerInfo] = []
        cut_settings = self._parse_cut_settings(project)

        for cs in cut_settings.values():
            name = cs.get("name", f"Layer {cs['index']}")
            layers.append(
                LayerInfo(
                    id=str(cs["index"]),
                    name=name,
                    feature_count=cs.get("shape_count", 0),
                )
            )

        return ImportManifest(
            title=title,
            layers=layers,
            natural_size_mm=None,
            warnings=self._warnings,
            errors=self._errors,
        )

    def _parse_cut_settings(
        self, project: ET.Element
    ) -> Dict[int, Dict[str, Any]]:
        cut_settings: Dict[int, Dict[str, Any]] = {}
        for cs_elem in list(project.findall("CutSetting")) + list(
            project.findall("CutSetting_Img")
        ):
            index_el = cs_elem.find("index")
            if index_el is None:
                continue
            idx = int(index_el.get("Value", "0"))
            params: Dict[str, Any] = {"index": idx}
            for child in cs_elem:
                tag = child.tag
                val = child.get("Value")
                if val is not None:
                    try:
                        if "." in val or "e" in val.lower():
                            params[tag] = float(val)
                        else:
                            params[tag] = int(val)
                    except ValueError:
                        params[tag] = val
                else:
                    params[tag] = child.text or ""
            cut_settings[idx] = params
        return cut_settings

    def _render_bitmaps_to_svg(self) -> Optional[bytes]:
        if not self._bitmaps:
            return None

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        image_tags: List[str] = []

        for bm in self._bitmaps:
            hw, hh = bm.width / 2.0, bm.height / 2.0
            a, b, c, d, tx, ty = bm.xform.for_cairo()
            data_url = "data:image/png;base64," + base64.b64encode(
                bm.png_data
            ).decode("ascii")
            transform = f"matrix({a} {-b} {c} {-d} {tx} {-ty})"

            corners_lx = [-hw, hw, hw, -hw]
            corners_ly = [-hh, -hh, hh, hh]
            for lx, ly in zip(corners_lx, corners_ly):
                sx = a * lx + c * ly + tx
                sy = -(b * lx + d * ly + ty)
                min_x = min(min_x, sx)
                min_y = min(min_y, sy)
                max_x = max(max_x, sx)
                max_y = max(max_y, sy)

            image_tags.append(
                f'<image x="{-hw}" y="{-hh}" '
                f'width="{bm.width}" height="{bm.height}" '
                f'transform="{transform}" '
                f'xlink:href="{data_url}"/>'
            )

        if min_x == float("inf"):
            return None

        vw = max(max_x - min_x, 1.0)
        vh = max(max_y - min_y, 1.0)
        svg_parts = [
            '<svg xmlns="http://www.w3.org/2000/svg"'
            ' xmlns:xlink="http://www.w3.org/1999/xlink"'
            f' viewBox="{min_x} {min_y} {vw} {vh}">'
        ]
        svg_parts.extend(image_tags)
        svg_parts.append("</svg>")
        return "".join(svg_parts).encode("utf-8")

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        _, _, w, h = parse_result.document_bounds

        merged = Geometry()
        for geo in self._geometries_by_layer.values():
            if geo:
                merged.extend(geo)
        thumbnail_data = (
            render_geometry_to_png(
                merged,
                256,
                line_width=2.0,
                color=(0.2, 0.2, 0.2, 1.0),
            )
            if not merged.is_empty()
            else None
        )

        # Store LightBurn cut settings in source asset metadata so they
        # survive project save/load and can be re-applied on re-import.
        cut_settings_by_name: Dict[str, Dict[str, Any]] = {}
        for idx, cs in self._cut_settings.items():
            name = cs.get("name", str(idx))
            cut_settings_by_name[name] = dict(cs)

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=LIGHTBURN_RENDERER,
            thumbnail_data=thumbnail_data,
            width_mm=w,
            height_mm=h,
        )
        source.metadata["lightburn_cut_settings"] = cut_settings_by_name

        if self._bitmaps:
            svg_data = self._render_bitmaps_to_svg()
            if svg_data:
                source.base_render_data = svg_data

        return source

    def _trace_bitmaps(
        self,
        parse_result: ParsingResult,
        spec: TraceSpec,
    ) -> Dict[Optional[str], Geometry]:
        geometries_by_layer: Dict[Optional[str], Geometry] = {}
        for bm in self._bitmaps:
            layer_id = str(bm.cut_index)
            try:
                img = pyvips.Image.pngload_buffer(
                    bm.png_data, access=pyvips.Access.SEQUENTIAL
                )
            except pyvips.Error:
                logger.warning("Failed to load bitmap for tracing")
                continue

            normalized = util.normalize_to_rgba(img)
            if normalized is None:
                continue

            surface = util.vips_rgba_to_cairo_surface(normalized)
            traced_geom_list = trace_surface(surface, spec)

            traced_geo = Geometry()
            for g in traced_geom_list:
                traced_geo.extend(g)

            if traced_geo.is_empty():
                continue

            traced_geo = _apply_xform_to_geo(traced_geo, bm.xform)
            if layer_id not in geometries_by_layer:
                geometries_by_layer[layer_id] = Geometry()
            geometries_by_layer[layer_id].extend(traced_geo)

        return geometries_by_layer

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        from ...core.vectorization_spec import (
            LayerImportMode,
            PassthroughSpec,
        )

        if isinstance(spec, TraceSpec):
            traced = self._trace_bitmaps(parse_result, spec)
            merged = Geometry()
            for geo in traced.values():
                merged.extend(geo)
            return VectorizationResult(
                geometries_by_layer=traced or {None: merged},
                source_parse_result=parse_result,
            )

        split_layers = False
        active_layers_set = None
        if isinstance(spec, PassthroughSpec):
            split_layers = spec.layer_import_mode != LayerImportMode.FLATTEN
            if spec.active_layer_ids:
                active_layers_set = set(spec.active_layer_ids)

        geometries: Dict[Optional[str], Geometry]
        if active_layers_set:
            geometries = {
                layer_id: geo
                for layer_id, geo in self._geometries_by_layer.items()
                if layer_id in active_layers_set
            }
        else:
            g: Dict[Optional[str], Geometry] = {}
            for k, v in self._geometries_by_layer.items():
                g[k] = v
            geometries = g

        merged_geo = Geometry()
        for geo in geometries.values():
            merged_geo.extend(geo)

        if split_layers:
            final_geometries: Dict[Optional[str], Geometry] = geometries or {
                None: merged_geo
            }
        else:
            final_geometries = {None: merged_geo}

        # Build per-layer settings from cut settings for the assembler.
        layer_settings: Dict[Optional[str], Dict[str, Any]] = {}
        for layer_id in final_geometries:
            if layer_id is None:
                continue
            try:
                cs = self._cut_settings.get(int(layer_id))
            except (ValueError, TypeError):
                cs = None
            if cs is None:
                continue
            config = _build_step_config(cs)
            if config:
                layer_settings[layer_id] = config

        return VectorizationResult(
            geometries_by_layer=final_geometries,
            source_parse_result=parse_result,
            layer_settings=layer_settings,
        )

    def parse(self) -> Optional[ParsingResult]:
        try:
            root = ET.fromstring(self.raw_data)
        except ET.ParseError as e:
            self.add_error(_(f"LightBurn file is corrupt or invalid: {e}"))
            return None

        project = root.find("LightBurnProject")
        if project is None:
            project = root

        self._cut_settings = self._parse_cut_settings(project)
        self._bitmaps = []

        geometries_by_layer: Dict[str, Geometry] = {}
        shape_count_by_layer: Dict[int, int] = {}

        for shape_elem in project.findall("Shape"):
            result = _shape_to_geometry(
                shape_elem, self._cut_settings, self._bitmaps
            )
            if result is not None:
                cut_idx, geo = result
                layer_id = str(cut_idx)
                if layer_id not in geometries_by_layer:
                    geometries_by_layer[layer_id] = Geometry()
                geometries_by_layer[layer_id].extend(geo)
                shape_count_by_layer[cut_idx] = (
                    shape_count_by_layer.get(cut_idx, 0) + 1
                )

        self._geometries_by_layer = geometries_by_layer

        for idx, cs in self._cut_settings.items():
            cs["shape_count"] = shape_count_by_layer.get(idx, 0)

        all_geo = Geometry()
        for geo in geometries_by_layer.values():
            if geo:
                all_geo.extend(geo)

        if all_geo.is_empty():
            document_bounds = (0.0, 0.0, 0.0, 0.0)
        else:
            min_x, min_y, max_x, max_y = all_geo.rect()
            w = max(max_x - min_x, 1e-9)
            h = max(max_y - min_y, 1e-9)
            document_bounds = (min_x, min_y, w, h)

        temp_result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=document_bounds,
            background_world_transform=None,  # type: ignore
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            document_bounds, temp_result
        )

        layer_geometries: List[LayerGeometry] = []
        for layer_id, geo in geometries_by_layer.items():
            if geo.is_empty():
                continue
            min_x, min_y, max_x, max_y = geo.rect()
            w = max(max_x - min_x, 1e-9)
            h = max(max_y - min_y, 1e-9)
            cs = self._cut_settings.get(int(layer_id), {})
            name = cs.get("name", layer_id)
            layer_geometries.append(
                LayerGeometry(
                    layer_id=layer_id,
                    name=str(name),
                    content_bounds=(min_x, min_y, w, h),
                )
            )

        return ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=layer_geometries,
            world_frame_of_reference=document_bounds,
            background_world_transform=bg_item.world_matrix,
        )
