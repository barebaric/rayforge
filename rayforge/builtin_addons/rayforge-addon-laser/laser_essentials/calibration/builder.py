"""
Turns a :class:`CalibrationLayout` into workpieces, steps and assets.

The builder creates the items but does not add them to the document;
their source assets are hidden, since they belong to the test and are
replaced when it is rebuilt;
``CalibrationCmd`` does that in one undoable transaction. Items are
created at their test-local position and then moved together, so the
conversion from test-local to world coordinates happens in one place
(:meth:`BuiltTest.move_to`).
"""

import logging
from dataclasses import dataclass, field
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.source_asset import SourceAsset
from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from rayforge.core.vectorization_spec import PassthroughSpec, ProceduralSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.procedural import ProceduralImporter
from rayforge.image.svg.exporter import GeometrySvgExporter
from rayforge.image.svg.importer import SvgImporter

from .layout import (
    CalibrationLayout,
    CellEntry,
    CommandEntry,
    FilledRect,
    ProceduralShape,
    Shape,
)

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

logger = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]


@dataclass
class BuiltTest:
    """The items of one test, before they are added to the document."""

    workpieces: list[WorkPiece] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    assets: list[SourceAsset] = field(default_factory=list)

    def bounds(self) -> Rect:
        """The world bounding box (min x, min y, max x, max y)."""
        return bounds_of(self.workpieces)

    def move_to(self, x: float, y: float) -> None:
        """Moves the test so its bottom-left corner is at (x, y)."""
        min_x, min_y, _max_x, _max_y = self.bounds()
        dx, dy = x - min_x, y - min_y
        for wp in self.workpieces:
            wp.pos = (wp.pos[0] + dx, wp.pos[1] + dy)

    def keep_inside(self, width: float, height: float) -> None:
        """
        Moves the test back onto an area of *width* × *height* mm where
        it sticks out, as far as it fits.
        """
        min_x, min_y, max_x, max_y = self.bounds()
        x = max(0.0, min(min_x, width - (max_x - min_x)))
        y = max(0.0, min(min_y, height - (max_y - min_y)))
        self.move_to(x, y)

    def center_on(self, width: float, height: float) -> None:
        """Centers the test on an area of *width* × *height* mm."""
        min_x, min_y, max_x, max_y = self.bounds()
        self.move_to(
            (width - (max_x - min_x)) / 2, (height - (max_y - min_y)) / 2
        )


def bounds_of(workpieces: list[WorkPiece]) -> Rect:
    rects = []
    for wp in workpieces:
        world = wp.get_world_geometry()
        if world is not None and not world.is_empty():
            rects.append(world.rect())
        else:
            x, y, width, height = wp.bbox
            rects.append((x, y, x + width, y + height))
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def filled_rect_svg(width: float, height: float) -> bytes:
    """An SVG of a black rectangle of *width* × *height* mm."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}mm" '
        f'height="{height:.3f}mm" viewBox="0 0 {width:.3f} {height:.3f}">'
        f'<rect width="{width:.3f}" height="{height:.3f}" fill="black"/>'
        f"</svg>"
    ).encode()


class CalibrationBuilder:
    """Creates document items for a calibration layout."""

    def __init__(self, context: "RayforgeContext"):
        self._context = context

    def build(self, layout: CalibrationLayout) -> BuiltTest:
        built = BuiltTest()
        for entry in layout.entries:
            if isinstance(entry, CommandEntry):
                built.steps.append(self._command_step(entry))
                continue
            assert isinstance(entry, CellEntry)
            source, wp = self._workpiece(entry.name, entry.shape)
            entry.step.workpiece_uids = [wp.uid]
            built.workpieces.append(wp)
            built.steps.append(entry.step)
            built.assets.append(source)
        return built

    def _command_step(self, entry: CommandEntry) -> Step:
        step_cls: Any = step_registry.get("CommandStep")
        if step_cls is None:
            raise ValueError(_("Machine commands need the Automation addon."))
        step = step_cls.create(self._context, name=entry.name)
        step.set_command_text("\n".join(entry.lines))
        return step

    def _workpiece(
        self, name: str, shape: Shape
    ) -> tuple[SourceAsset, WorkPiece]:
        if isinstance(shape, ProceduralShape):
            source, wp, target = self._procedural(name, shape)
        elif isinstance(shape, FilledRect):
            data = filled_rect_svg(shape.width, shape.height)
            source, wp = self._import_svg(name, data)
            target = (shape.x, shape.y)
        else:
            data = GeometrySvgExporter(shape.geometry).export()
            source, wp = self._import_svg(name, data)
            target = shape.geometry.rect()[:2]
        source.name = name
        source.hidden = True
        wp.name = name
        min_x, min_y, _max_x, _max_y = bounds_of([wp])
        wp.pos = (
            wp.pos[0] + target[0] - min_x,
            wp.pos[1] + target[1] - min_y,
        )
        return source, wp

    @staticmethod
    def _import_svg(name: str, data: bytes) -> tuple[SourceAsset, WorkPiece]:
        result = SvgImporter(data).get_doc_items(PassthroughSpec())
        if not result or not result.payload or not result.payload.items:
            raise RuntimeError(f"Could not create the {name} workpiece.")
        wp = result.payload.items[0]
        assert isinstance(wp, WorkPiece)
        return result.payload.source, wp

    @staticmethod
    def _procedural(
        name: str, shape: ProceduralShape
    ) -> tuple[SourceAsset, WorkPiece, tuple[float, float]]:
        draw = shape.draw_function
        size = shape.size_function
        importer = ProceduralImporter(
            drawing_function_path=f"{draw.__module__}.{draw.__name__}",
            size_function_path=f"{size.__module__}.{size.__name__}",
            params=shape.params,
            name=name,
        )
        result = importer.get_doc_items(ProceduralSpec())
        if not result or not result.payload or not result.payload.items:
            raise RuntimeError(f"Could not create the {name} workpiece.")
        wp = result.payload.items[0]
        assert isinstance(wp, WorkPiece)
        return result.payload.source, wp, shape.origin
