import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...image.hull import get_concave_hull
from ...image.tracing import prepare_surface
from ...core.matrix import Matrix
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
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError(
                "ShrinkWrapProducer requires a workpiece context."
            )

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        boolean_image = prepare_surface(surface)

        if np.any(boolean_image):
            # The hull is generated in pixel coordinates.
            geometry = get_concave_hull(
                boolean_image=boolean_image,
                scale_x=1.0,  # Generate in pixels
                scale_y=1.0,  # Generate in pixels
                height_px=surface.get_height(),
                border_size=BORDER_SIZE,
                gravity=self.gravity,
            )

            if geometry:
                hull_ops = Ops.from_geometry(geometry)

                # Scale the pixel-based ops to their final millimeter size.
                width_mm, height_mm = workpiece.size
                px_width, px_height = surface.get_width(), surface.get_height()

                if px_width > 0 and px_height > 0:
                    scale_x = width_mm / px_width
                    scale_y = height_mm / px_height
                    scaling_matrix = Matrix.scale(scale_x, scale_y)
                    hull_ops.transform(scaling_matrix.to_4x4_numpy())

                final_ops.extend(hull_ops)

        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        return PipelineArtifact(
            ops=final_ops,
            is_scalable=True,  # Source data is vector, so it's scalable
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

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
