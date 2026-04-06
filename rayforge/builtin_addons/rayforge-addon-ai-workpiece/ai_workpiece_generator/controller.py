import asyncio
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, cast

from svgelements import SVG, Path as SvgPath

from rayforge.core.asset_registry import asset_type_registry
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.image.svg.svgutil import PPI
from rayforge.shared.tasker import task_mgr
from .generator import generate_svg

if TYPE_CHECKING:
    from rayforge.core.geometry_provider import IGeometryProvider


class SketchInstanceProtocol(Protocol):
    """Protocol for Sketch instance with name attribute."""

    name: str


class SketchClassProtocol(Protocol):
    """Protocol for Sketch class with from_geometry classmethod."""

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> SketchInstanceProtocol:
        """Create a Sketch from Geometry."""
        ...


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of AI generation with optional sketch conversion."""

    sketch: Optional["IGeometryProvider"] = None
    svg_content: Optional[str] = None
    geometry: Optional[Geometry] = None
    error: Optional[str] = None


def parse_svg_to_geometry(svg_content: str) -> Geometry:
    """
    Parse SVG content to Geometry.

    Returns Geometry in SVG user units (mm if properly configured).
    """
    svg_stream = io.BytesIO(svg_content.encode("utf-8"))
    svg = SVG.parse(svg_stream, ppi=PPI)

    geo = Geometry()
    for shape in svg.elements():
        try:
            path = SvgPath(shape)
            path.reify()
            _add_path_to_geometry(path, geo)
        except (AttributeError, TypeError):
            continue

    if not geo.is_empty():
        geo = _convert_pixel_to_user_units(svg, geo)

    return geo


def _convert_pixel_to_user_units(svg: SVG, geo_px: Geometry) -> Geometry:
    """Convert geometry from pixel coordinates to SVG user units."""
    w_px = _length_to_px(svg.width)
    h_px = _length_to_px(svg.height)

    if svg.viewbox:
        vb_x = _to_float(svg.viewbox.x)
        vb_y = _to_float(svg.viewbox.y)
        vb_w = _to_float(svg.viewbox.width, default=w_px)
        vb_h = _to_float(svg.viewbox.height, default=h_px)
    else:
        vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, w_px, h_px

    scale_x = vb_w / w_px if w_px > 0 else 1.0
    scale_y = vb_h / h_px if h_px > 0 else 1.0

    transform = Matrix.translation(vb_x, vb_y) @ Matrix.scale(scale_x, scale_y)

    geo_user = geo_px.copy()
    geo_user.transform(transform.to_4x4_numpy())
    return geo_user


def _length_to_px(value) -> float:
    """Convert a length value to pixels."""
    if value is not None:
        try:
            return float(getattr(value, "px", value))
        except (ValueError, TypeError):
            return 1.0
    return 1.0


def _to_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    return float(value) if value is not None else default


def _add_path_to_geometry(path: SvgPath, geo: Geometry) -> None:
    """Add SVG path segments to Geometry."""
    from svgelements import (
        Close,
        Move,
        Line,
        CubicBezier,
        QuadraticBezier,
        Arc,
    )

    for seg in path:
        end_pt = (0.0, 0.0)
        if not isinstance(seg, Close):
            if seg.end is None or seg.end.x is None or seg.end.y is None:
                continue
            end_pt = (float(seg.end.x), float(seg.end.y))

        if isinstance(seg, Move):
            geo.move_to(end_pt[0], end_pt[1])
        elif isinstance(seg, Line):
            geo.line_to(end_pt[0], end_pt[1])
        elif isinstance(seg, Close):
            geo.close_path()
        elif isinstance(seg, CubicBezier):
            if (
                seg.control1 is not None
                and seg.control1.x is not None
                and seg.control1.y is not None
                and seg.control2 is not None
                and seg.control2.x is not None
                and seg.control2.y is not None
            ):
                c1 = (float(seg.control1.x), float(seg.control1.y))
                c2 = (float(seg.control2.x), float(seg.control2.y))
                geo.bezier_to(end_pt[0], end_pt[1], c1[0], c1[1], c2[0], c2[1])
            else:
                geo.line_to(end_pt[0], end_pt[1])
        elif isinstance(seg, QuadraticBezier):
            if (
                seg.start is not None
                and seg.start.x is not None
                and seg.start.y is not None
                and seg.control is not None
                and seg.control.x is not None
                and seg.control.y is not None
            ):
                sx, sy = float(seg.start.x), float(seg.start.y)
                cx, cy = float(seg.control.x), float(seg.control.y)
                ex, ey = end_pt
                c1x = sx + (2.0 / 3.0) * (cx - sx)
                c1y = sy + (2.0 / 3.0) * (cy - sy)
                c2x = ex + (2.0 / 3.0) * (cx - ex)
                c2y = ey + (2.0 / 3.0) * (cy - ey)
                geo.bezier_to(ex, ey, c1x, c1y, c2x, c2y)
            else:
                geo.line_to(end_pt[0], end_pt[1])
        elif isinstance(seg, Arc):
            for cubic in seg.as_cubic_curves():
                if (
                    cubic.end is not None
                    and cubic.end.x is not None
                    and cubic.end.y is not None
                    and cubic.control1 is not None
                    and cubic.control1.x is not None
                    and cubic.control1.y is not None
                    and cubic.control2 is not None
                    and cubic.control2.x is not None
                    and cubic.control2.y is not None
                ):
                    e = (float(cubic.end.x), float(cubic.end.y))
                    c1 = (float(cubic.control1.x), float(cubic.control1.y))
                    c2 = (float(cubic.control2.x), float(cubic.control2.y))
                    geo.bezier_to(e[0], e[1], c1[0], c1[1], c2[0], c2[1])
                elif (
                    cubic.end is not None
                    and cubic.end.x is not None
                    and cubic.end.y is not None
                ):
                    geo.line_to(float(cubic.end.x), float(cubic.end.y))


class AISvgGeneratorController:
    """Controller for AI SVG generation - pure business logic."""

    def __init__(self):
        self._cancelled = False

    def generate(
        self,
        prompt: str,
        on_success: Callable[[GenerationResult], None],
        on_error: Callable[[str], None],
    ) -> None:
        """
        Generate SVG and attempt to convert to editable Sketch.

        Args:
            prompt: The text prompt for SVG generation
            on_success: Callback with GenerationResult containing sketch or
                        svg_content for fallback
            on_error: Callback with error message
        """
        self._cancelled = False

        async def do_generate():
            try:
                svg_content, error = await generate_svg(prompt)

                if self._cancelled:
                    return

                if error:
                    task_mgr.schedule_on_main_thread(on_error, error)
                    return

                if not svg_content:
                    task_mgr.schedule_on_main_thread(
                        on_error, "Failed to generate SVG."
                    )
                    return

                result = GenerationResult(svg_content=svg_content)

                try:
                    geometry = parse_svg_to_geometry(svg_content)
                    result.geometry = geometry

                    if not geometry.is_empty():
                        sketch_cls = asset_type_registry.get("sketch")
                        if sketch_cls:
                            sketch = cast(
                                SketchClassProtocol, sketch_cls
                            ).from_geometry(geometry)
                            sketch.name = prompt[:50]
                            result.sketch = cast("IGeometryProvider", sketch)
                        logger.info(
                            "Successfully converted AI-generated SVG to "
                            "editable sketch"
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to convert SVG to sketch: %s", e, exc_info=True
                    )

                if self._cancelled:
                    return

                task_mgr.schedule_on_main_thread(on_success, result)

            except Exception as e:
                logger.error("Error in generation task: %s", e, exc_info=True)
                if not self._cancelled:
                    task_mgr.schedule_on_main_thread(on_error, str(e))

        future = asyncio.run_coroutine_threadsafe(do_generate(), task_mgr.loop)

        def on_done(f):
            try:
                f.result()
            except Exception as e:
                logger.error("Generation future error: %s", e)
                if not self._cancelled:
                    task_mgr.schedule_on_main_thread(on_error, str(e))

        future.add_done_callback(on_done)

    def cancel(self) -> None:
        """Cancel any ongoing generation."""
        self._cancelled = True


controller = AISvgGeneratorController()
