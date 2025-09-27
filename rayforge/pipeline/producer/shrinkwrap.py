import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING
from .base import OpsProducer
from ...image.hull import get_concave_hull
from ...image.tracing import prepare_surface
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece

BORDER_SIZE = 2


class ShrinkWrapProducer(OpsProducer):
    """
    Generates a single vector hull that encloses all content on a surface.

    This producer supports a "gravity" setting, which creates a concave hull
    that "shrink-wraps" the content. A gravity of 0.0 will produce a
    standard convex hull.
    """

    def __init__(self, gravity: float = 0.0):
        """
        Initializes the HullProducer.

        Args:
            gravity: A factor from 0.0 to 1.0. 0.0 results in a normal convex
                     hull. Higher values increase the "shrink-wrap" effect.
        """
        self.gravity = gravity

    def run(
        self,
        laser,
        surface: cairo.ImageSurface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> Ops:
        if workpiece is None:
            raise ValueError("HullProducer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        boolean_image = prepare_surface(surface)

        if np.any(boolean_image):
            if pixels_per_mm:
                px_per_mm_x, px_per_mm_y = pixels_per_mm
            else:
                # We are in scalable mode, work in pixel coordinates
                px_per_mm_x, px_per_mm_y = 1.0, 1.0

            height_px = surface.get_height()

            # get_concave_hull handles both convex (gravity=0) and concave
            # cases, creating the "rubber band with gravity" effect.
            geometry = get_concave_hull(
                boolean_image=boolean_image,
                scale_x=px_per_mm_x,
                scale_y=px_per_mm_y,
                height_px=height_px,
                border_size=BORDER_SIZE,
                gravity=self.gravity,
            )

            if geometry:
                hull_ops = Ops.from_geometry(geometry)
                final_ops.extend(hull_ops)

        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
        return final_ops

    @property
    def requires_full_render(self) -> bool:
        """
        Overrides the base property to signal that this producer must receive
        the entire rendered workpiece as a raster image, even though its
        output is scalable.
        """
        return True

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {"gravity": self.gravity},
        }
