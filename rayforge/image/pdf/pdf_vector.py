from __future__ import annotations
import logging
import re
import math
from typing import Optional, Dict, Any, List, Tuple, cast
from pathlib import Path
from gettext import gettext as _

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from ...core.geo import Geometry
from ...core.matrix import Matrix
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
    VectorizationSpec,
)
from ..base_importer import Importer, ImporterFeature
from ..structures import (
    ParsingResult,
    LayerGeometry,
    LayerInfo,
    VectorizationResult,
    ImportManifest,
)
from ..util import to_mm
from .renderer import PDF_RENDERER
from ..engine import NormalizationEngine

logger = logging.getLogger(__name__)

PT_TO_MM = 25.4 / 72.0


class PdfVectorImporter(Importer):
    """
    Imports vector data directly from PDF files using pymupdf.

    Extracts vector paths (lines, curves, shapes) from PDF pages without
    rasterization, preserving the original vector geometry.
    """

    label = "PDF (Vector Strategy)"
    mime_types = ()
    extensions = ()
    features = {ImporterFeature.DIRECT_VECTOR, ImporterFeature.LAYER_SELECTION}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._doc: Optional[pymupdf.Document] = None
        self._page: Optional[pymupdf.Page] = None
        self._page_width_pt: float = 0.0
        self._page_height_pt: float = 0.0
        self._geometries_by_layer: Dict[Optional[str], Geometry] = {}

    def scan(self) -> ImportManifest:
        try:
            doc = pymupdf.open(stream=self.raw_data, filetype="pdf")
            if doc.page_count == 0:
                doc.close()
                self.add_error(_("PDF file contains no pages."))
                return ImportManifest(
                    title=self.source_file.name, errors=self._errors
                )

            page = doc[0]
            mediabox = page.mediabox
            width_pt = float(mediabox.width)
            height_pt = float(mediabox.height)
            size_mm = (to_mm(width_pt, "pt"), to_mm(height_pt, "pt"))

            title = (doc.metadata or {}).get("title") or self.source_file.name

            ocgs = doc.get_ocgs()
            layers = []
            if ocgs:
                for ocg_id, ocg_info in ocgs.items():
                    layer_name = ocg_info.get("name", str(ocg_id))
                    layers.append(
                        LayerInfo(
                            id=layer_name,
                            name=layer_name,
                            default_active=ocg_info.get("on", True),
                        )
                    )

            doc.close()

            return ImportManifest(
                title=title,
                layers=layers,
                natural_size_mm=size_mm,
                warnings=self._warnings,
                errors=self._errors,
            )
        except Exception as e:
            logger.warning(f"PDF scan failed for {self.source_file.name}: {e}")
            self.add_error(_(f"Could not read PDF: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

    def parse(self) -> Optional[ParsingResult]:
        try:
            self._doc = pymupdf.open(stream=self.raw_data, filetype="pdf")
            if self._doc.page_count == 0:
                self.add_error(_("PDF file contains no pages."))
                self._close_document()
                return None

            self._page = self._doc[0]
            mediabox = self._page.mediabox
            self._page_width_pt = float(mediabox.width)
            self._page_height_pt = float(mediabox.height)

            if self._page_width_pt <= 0 or self._page_height_pt <= 0:
                self.add_error(_("PDF page has zero dimensions"))
                self._close_document()
                return None

            document_bounds = (
                0.0,
                0.0,
                self._page_width_pt,
                self._page_height_pt,
            )

            native_unit_to_mm = PT_TO_MM
            x, y, w, h = document_bounds
            world_frame = (
                x * native_unit_to_mm,
                0.0,
                w * native_unit_to_mm,
                h * native_unit_to_mm,
            )

            temp_result = ParsingResult(
                document_bounds=document_bounds,
                native_unit_to_mm=native_unit_to_mm,
                is_y_down=True,
                layers=[],
                world_frame_of_reference=world_frame,
                background_world_transform=Matrix(),
            )

            bg_item = NormalizationEngine.calculate_layout_item(
                document_bounds, temp_result
            )

            geometries = self._extract_page_geometry()
            self._geometries_by_layer = geometries

            if not geometries or all(
                g.is_empty() for g in geometries.values()
            ):
                self.add_warning(_("PDF contains no vector geometry."))

            layer_geometries: List[LayerGeometry] = []
            for layer_id, geo in geometries.items():
                if not geo.is_empty():
                    layer_geometries.append(
                        LayerGeometry(
                            layer_id=layer_id or "__default__",
                            name=layer_id or "__default__",
                            content_bounds=document_bounds,
                        )
                    )

            if not layer_geometries:
                layer_geometries.append(
                    LayerGeometry(
                        layer_id="__default__",
                        name="__default__",
                        content_bounds=document_bounds,
                    )
                )

            return ParsingResult(
                document_bounds=document_bounds,
                native_unit_to_mm=native_unit_to_mm,
                is_y_down=True,
                layers=layer_geometries,
                world_frame_of_reference=world_frame,
                background_world_transform=bg_item.world_matrix,
            )

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}", exc_info=True)
            self.add_error(_(f"Failed to parse PDF: {e}"))
            self._close_document()
            return None

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        if not isinstance(spec, PassthroughSpec):
            spec = PassthroughSpec()

        if self._page is None:
            logger.error("vectorize() called before parse()")
            return VectorizationResult(
                geometries_by_layer={}, source_parse_result=parse_result
            )

        split_layers = False
        active_layers_set = None
        if isinstance(spec, PassthroughSpec):
            split_layers = spec.layer_import_mode != LayerImportMode.FLATTEN
            if spec.active_layer_ids:
                active_layers_set = set(spec.active_layer_ids)

        geometries: Dict[Optional[str], Geometry] = self._geometries_by_layer
        if not geometries:
            geometries = {None: Geometry()}

        geometries_to_process: Dict[Optional[str], Geometry]
        if active_layers_set:
            geometries_to_process = {
                layer_id: geo
                for layer_id, geo in geometries.items()
                if layer_id in active_layers_set
            }
        else:
            geometries_to_process = geometries

        final_geometries: Dict[Optional[str], Geometry]
        if split_layers:
            final_geometries = {}
            for layer_id, geo in geometries_to_process.items():
                final_layer_id = layer_id or "__default__"
                final_geometries[final_layer_id] = geo
        else:
            merged_geo = Geometry()
            for geo in geometries_to_process.values():
                merged_geo.extend(geo)
            final_geometries = {"__default__": merged_geo}

        self._close_document()

        return VectorizationResult(
            geometries_by_layer=final_geometries,
            source_parse_result=parse_result,
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        width_mm = self._page_width_pt * parse_result.native_unit_to_mm
        height_mm = self._page_height_pt * parse_result.native_unit_to_mm

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PDF_RENDERER,
            thumbnail_data=self._render_thumbnail_from_renderer(
                PDF_RENDERER, self.raw_data
            ),
            width_px=int(self._page_width_pt),
            height_px=int(self._page_height_pt),
            width_mm=width_mm,
            height_mm=height_mm,
        )

        return source

    def _close_document(self) -> None:
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None
            self._page = None

    def _extract_page_geometry(self) -> Dict[Optional[str], Geometry]:
        if self._page is None:
            return {None: Geometry()}

        geometries_by_layer: Dict[Optional[str], Geometry] = {}

        try:
            drawings = self._page.get_drawings()
            for drawing in drawings:
                layer_name: Optional[str] = drawing.get("layer")
                if layer_name not in geometries_by_layer:
                    geometries_by_layer[layer_name] = Geometry()
                self._add_drawing_to_geometry(
                    drawing, geometries_by_layer[layer_name]
                )
        except Exception as e:
            logger.warning(f"Failed to extract drawings: {e}")

        if not geometries_by_layer:
            geometries_by_layer[None] = Geometry()

        return geometries_by_layer

    def _add_drawing_to_geometry(
        self, drawing: Dict[str, Any], geometry: Geometry
    ) -> None:
        items = cast(List[tuple], drawing.get("items", []))
        if not items:
            return

        dashes_str = cast(str, drawing.get("dashes", ""))
        pattern, phase = self._parse_dash_pattern(dashes_str)
        if pattern:
            items = self._expand_dashed_items(items, pattern, phase)

        first_cmd = items[0][0] if items else None
        if first_cmd != "m":
            start_pt = self._get_start_point(items[0])
            if start_pt is not None:
                geometry.move_to(float(start_pt.x), float(start_pt.y))

        last_end_pt: Optional[pymupdf.Point] = None
        for item in items:
            if not isinstance(item, tuple) or len(item) < 1:
                continue

            cmd = item[0]

            if cmd == "m":
                if len(item) >= 2 and isinstance(item[1], pymupdf.Point):
                    pt = item[1]
                    geometry.move_to(float(pt.x), float(pt.y))
                    last_end_pt = pt

            elif cmd == "l":
                if len(item) >= 3 and isinstance(item[2], pymupdf.Point):
                    start_pt = item[1]
                    end_pt = item[2]
                    if last_end_pt is not None and self._points_differ(
                        start_pt, last_end_pt
                    ):
                        geometry.move_to(float(start_pt.x), float(start_pt.y))
                    geometry.line_to(float(end_pt.x), float(end_pt.y))
                    last_end_pt = end_pt

            elif cmd == "c":
                if len(item) >= 5 and all(
                    isinstance(item[i], pymupdf.Point) for i in range(1, 5)
                ):
                    start_pt = item[1]
                    c1 = item[2]
                    c2 = item[3]
                    end_pt = item[4]
                    if last_end_pt is not None and self._points_differ(
                        start_pt, last_end_pt
                    ):
                        geometry.move_to(float(start_pt.x), float(start_pt.y))
                    geometry.bezier_to(
                        float(end_pt.x),
                        float(end_pt.y),
                        float(c1.x),
                        float(c1.y),
                        float(c2.x),
                        float(c2.y),
                    )
                    last_end_pt = end_pt

            elif cmd == "h":
                geometry.close_path()

            elif cmd == "re":
                if len(item) >= 2:
                    rect = item[1]
                    if hasattr(rect, "x0") and hasattr(rect, "y0"):
                        x = float(rect.x0)
                        y = float(rect.y0)
                        w = float(rect.width)
                        h = float(rect.height)
                        self._add_rect_to_geometry(geometry, x, y, w, h)
                        last_end_pt = None

            elif cmd == "q":
                pass

            elif cmd == "Q":
                pass

    def _points_differ(
        self, p1: pymupdf.Point, p2: pymupdf.Point, tolerance: float = 0.01
    ) -> bool:
        return (
            abs(float(p1.x) - float(p2.x)) > tolerance
            or abs(float(p1.y) - float(p2.y)) > tolerance
        )

    def _get_start_point(self, item: tuple) -> Optional[pymupdf.Point]:
        cmd = item[0] if item else None
        if (
            cmd == "l"
            and len(item) >= 2
            and isinstance(item[1], pymupdf.Point)
        ):
            return item[1]
        if (
            cmd == "c"
            and len(item) >= 2
            and isinstance(item[1], pymupdf.Point)
        ):
            return item[1]
        return None

    def _add_rect_to_geometry(
        self, geometry: Geometry, x: float, y: float, w: float, h: float
    ) -> None:
        geometry.move_to(x, y)
        geometry.line_to(x + w, y)
        geometry.line_to(x + w, y + h)
        geometry.line_to(x, y + h)
        geometry.close_path()

    def _parse_dash_pattern(
        self, dashes_str: str
    ) -> Tuple[List[float], float]:
        if not dashes_str or dashes_str == "[] 0":
            return [], 0.0
        match = re.match(r"\[\s*([\d.\s]+)\s*\]\s*(\d+\.?\d*)", dashes_str)
        if not match:
            return [], 0.0
        pattern = [float(x) for x in match.group(1).split()]
        phase = float(match.group(2))
        return pattern, phase

    def _expand_dashed_items(
        self, items: List[tuple], pattern: List[float], phase: float
    ) -> List[tuple]:
        if not pattern:
            return items
        expanded = []
        dash_pos = phase
        pattern_idx = 0
        for item in items:
            cmd = item[0]
            if cmd == "m":
                expanded.append(item)
                dash_pos = phase
                pattern_idx = 0
            elif cmd == "l" and len(item) >= 3:
                start_pt, end_pt = item[1], item[2]
                segments = self._dash_line(
                    float(start_pt.x),
                    float(start_pt.y),
                    float(end_pt.x),
                    float(end_pt.y),
                    pattern,
                    dash_pos,
                    pattern_idx,
                )
                expanded.extend(segments)
                length = math.hypot(
                    float(end_pt.x) - float(start_pt.x),
                    float(end_pt.y) - float(start_pt.y),
                )
                dash_pos, pattern_idx = self._advance_dash(
                    length, pattern, dash_pos, pattern_idx
                )
            elif cmd == "c" and len(item) >= 5:
                start_pt, c1, c2, end_pt = item[1], item[2], item[3], item[4]
                segments = self._dash_bezier(
                    float(start_pt.x),
                    float(start_pt.y),
                    float(c1.x),
                    float(c1.y),
                    float(c2.x),
                    float(c2.y),
                    float(end_pt.x),
                    float(end_pt.y),
                    pattern,
                    dash_pos,
                    pattern_idx,
                )
                expanded.extend(segments)
                length = self._bezier_arc_length(
                    float(start_pt.x),
                    float(start_pt.y),
                    float(c1.x),
                    float(c1.y),
                    float(c2.x),
                    float(c2.y),
                    float(end_pt.x),
                    float(end_pt.y),
                )
                dash_pos, pattern_idx = self._advance_dash(
                    length, pattern, dash_pos, pattern_idx
                )
            elif cmd == "h":
                expanded.append(item)
            elif cmd == "re" and len(item) >= 5:
                expanded.append(item)
        return expanded

    def _dash_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        pattern: List[float],
        dash_pos: float,
        pattern_idx: int,
    ) -> List[tuple]:
        segments = []
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 0.001:
            return segments
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        remaining = pattern[pattern_idx] - dash_pos
        is_on = (pattern_idx % 2) == 0
        while pos < length:
            if remaining <= 1e-6:
                pattern_idx = (pattern_idx + 1) % len(pattern)
                is_on = not is_on
                remaining = pattern[pattern_idx]
            segment_len = min(remaining, length - pos)
            if is_on:
                sx1 = x1 + dx * pos
                sy1 = y1 + dy * pos
                sx2 = x1 + dx * (pos + segment_len)
                sy2 = y1 + dy * (pos + segment_len)
                if not segments:
                    segments.append(("m", pymupdf.Point(sx1, sy1)))
                else:
                    last = segments[-1]
                    if last[0] == "l":
                        last_end = last[2]
                        if (
                            abs(float(last_end.x) - sx1) > 0.01
                            or abs(float(last_end.y) - sy1) > 0.01
                        ):
                            segments.append(("m", pymupdf.Point(sx1, sy1)))
                    else:
                        segments.append(("m", pymupdf.Point(sx1, sy1)))
                segments.append(
                    ("l", pymupdf.Point(sx1, sy1), pymupdf.Point(sx2, sy2))
                )
            pos += segment_len
            remaining -= segment_len
        return segments

    def _advance_dash(
        self,
        length: float,
        pattern: List[float],
        dash_pos: float,
        pattern_idx: int,
    ) -> Tuple[float, int]:
        if not pattern:
            return 0.0, 0
        total_pos = dash_pos + length
        while total_pos >= pattern[pattern_idx]:
            total_pos -= pattern[pattern_idx]
            pattern_idx = (pattern_idx + 1) % len(pattern)
        return total_pos, pattern_idx

    def _bezier_arc_length(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        steps: int = 20,
    ) -> float:
        length = 0.0
        prev_x, prev_y = x0, y0
        for i in range(1, steps + 1):
            t = i / steps
            t2, t3 = t * t, t * t * t
            mt, mt2, mt3 = 1 - t, (1 - t) ** 2, (1 - t) ** 3
            px = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
            py = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
            length += math.hypot(px - prev_x, py - prev_y)
            prev_x, prev_y = px, py
        return length

    def _bezier_point(
        self,
        t: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
    ) -> Tuple[float, float]:
        t2, t3 = t * t, t * t * t
        mt, mt2, mt3 = 1 - t, (1 - t) ** 2, (1 - t) ** 3
        px = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
        py = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
        return px, py

    def _dash_bezier(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        pattern: List[float],
        dash_pos: float,
        pattern_idx: int,
    ) -> List[tuple]:
        segments = []
        total_length = self._bezier_arc_length(x0, y0, x1, y1, x2, y2, x3, y3)
        if total_length < 0.001:
            return segments
        steps = max(20, int(total_length / 5))
        arc_lengths = [0.0]
        prev_x, prev_y = x0, y0
        for i in range(1, steps + 1):
            t = i / steps
            px, py = self._bezier_point(t, x0, y0, x1, y1, x2, y2, x3, y3)
            arc_lengths.append(
                arc_lengths[-1] + math.hypot(px - prev_x, py - prev_y)
            )
            prev_x, prev_y = px, py

        def t_for_arc_len(target_len: float) -> float:
            for i in range(1, len(arc_lengths)):
                if arc_lengths[i] >= target_len:
                    frac = (target_len - arc_lengths[i - 1]) / (
                        arc_lengths[i] - arc_lengths[i - 1]
                    )
                    return (i - 1 + frac) / steps
            return 1.0

        pos = 0.0
        remaining = pattern[pattern_idx] - dash_pos
        is_on = (pattern_idx % 2) == 0
        while pos < total_length:
            if remaining <= 1e-6:
                pattern_idx = (pattern_idx + 1) % len(pattern)
                is_on = not is_on
                remaining = pattern[pattern_idx]
            segment_len = min(remaining, total_length - pos)
            if is_on:
                t_start = t_for_arc_len(pos)
                t_end = t_for_arc_len(pos + segment_len)
                sx, sy = self._bezier_point(
                    t_start, x0, y0, x1, y1, x2, y2, x3, y3
                )
                if not segments:
                    segments.append(("m", pymupdf.Point(sx, sy)))
                else:
                    last = segments[-1]
                    if last[0] in ("l", "c"):
                        last_end = last[2] if last[0] == "l" else last[4]
                        if (
                            abs(float(last_end.x) - sx) > 0.01
                            or abs(float(last_end.y) - sy) > 0.01
                        ):
                            segments.append(("m", pymupdf.Point(sx, sy)))
                    else:
                        segments.append(("m", pymupdf.Point(sx, sy)))
                if t_end - t_start >= 0.001:
                    ex, ey = self._bezier_point(
                        t_end, x0, y0, x1, y1, x2, y2, x3, y3
                    )
                    c1x, c1y = self._bezier_point(
                        t_start + (t_end - t_start) / 3,
                        x0,
                        y0,
                        x1,
                        y1,
                        x2,
                        y2,
                        x3,
                        y3,
                    )
                    c2x, c2y = self._bezier_point(
                        t_start + 2 * (t_end - t_start) / 3,
                        x0,
                        y0,
                        x1,
                        y1,
                        x2,
                        y2,
                        x3,
                        y3,
                    )
                    segments.append(
                        (
                            "c",
                            pymupdf.Point(sx, sy),
                            pymupdf.Point(c1x, c1y),
                            pymupdf.Point(c2x, c2y),
                            pymupdf.Point(ex, ey),
                        )
                    )
            pos += segment_len
            remaining -= segment_len
        return segments

    def __del__(self):
        self._close_document()
