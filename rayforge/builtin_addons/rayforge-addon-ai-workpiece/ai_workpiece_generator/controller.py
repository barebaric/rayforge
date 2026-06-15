import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, cast

from raygeo.geo import Geometry
from raygeo.svg import svg_string_to_geometry

from rayforge.core.asset_registry import asset_type_registry
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
                    geometry = svg_string_to_geometry(svg_content, 1.0, 1.0)
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
