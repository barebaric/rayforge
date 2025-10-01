import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING, Dict, Any
from .base import OpsProducer, PipelineArtifact, CoordinateSystem, CutSide
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
    from ...machine.models.laser import Laser

BORDER_SIZE = 2


class ShrinkWrapProducer(OpsProducer):
    """
    Generates a single vector hull that encloses all content on a surface.

    This producer supports a "gravity" setting, which creates a concave hull
    that "shrink-wraps" the content. A gravity of 0.0 will produce a
    standard convex hull.
    """

    def __init__(
        self,
        gravity: float = 0.0,
        path_offset_mm: float = 0.0,
        cut_side: CutSide = CutSide.OUTSIDE,
    ):
        """
        Initializes the producer.

        Args:
            gravity: A factor from 0.0 to 1.0. 0.0 results in a normal convex
                     hull. Higher values increase the "shrink-wrap" effect.
            path_offset_mm: An absolute distance to offset the generated path.
            cut_side: The rule for determining the final cut side. A hull is
                      typically an OUTSIDE cut.
        """
        super().__init__()
        self.gravity = gravity
        self.path_offset_mm = path_offset_mm
        self.cut_side = cut_side

    def run(
        self,
        laser: "Laser",
        surface: cairo.ImageSurface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError(
                "ShrinkWrapProducer requires a workpiece context."
            )

        final_ops = Ops()

        # 1. Calculate total offset
        kerf_mm = (settings or {}).get("kerf_mm", laser.spot_size_mm[0])
        kerf_compensation = kerf_mm / 2.0
        total_offset = 0.0
        if self.cut_side == CutSide.CENTERLINE:
            total_offset = 0.0  # Centerline ignores path offset
        elif self.cut_side == CutSide.OUTSIDE:
            total_offset = self.path_offset_mm + kerf_compensation
        elif self.cut_side == CutSide.INSIDE:
            total_offset = -self.path_offset_mm - kerf_compensation

        # 2. Generate base geometry in pixel space
        boolean_image = prepare_surface(surface)
        hull_geometry = None
        if np.any(boolean_image):
            hull_geometry = get_concave_hull(
                boolean_image=boolean_image,
                scale_x=1.0,
                scale_y=1.0,
                height_px=surface.get_height(),
                border_size=BORDER_SIZE,
                gravity=self.gravity,
            )

        if hull_geometry:
            # 3. Scale the pixel-based geometry to final millimeter size first
            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()
            if px_width > 0 and px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height
                scaling_matrix = Matrix.scale(scale_x, scale_y)
                hull_geometry.transform(scaling_matrix.to_4x4_numpy())

            # 4. Apply offset in millimeter space
            if abs(total_offset) > 1e-6:
                hull_geometry = hull_geometry.grow(total_offset)

            # 5. Convert to Ops
            final_ops.set_laser(laser.uid)
            final_ops.add(
                OpsSectionStartCommand(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
            )
            final_ops.set_power((settings or {}).get("power", 0))
            final_ops.extend(Ops.from_geometry(hull_geometry))
            final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        # 6. Create the artifact. The ops are pre-scaled, so they are not
        #    scalable in the pipeline cache sense.
        return PipelineArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

    @property
    def supports_kerf(self) -> bool:
        return True

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
            "params": {
                "gravity": self.gravity,
                "path_offset_mm": self.path_offset_mm,
                "cut_side": self.cut_side.name,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShrinkWrapProducer":
        """Deserializes a dictionary into a ShrinkWrapProducer instance."""
        params = data.get("params", {})
        cut_side_str = params.get(
            "cut_side", params.get("kerf_mode", "OUTSIDE")
        )
        try:
            cut_side = CutSide[cut_side_str]
        except KeyError:
            cut_side = CutSide.OUTSIDE

        return cls(
            gravity=params.get("gravity", 0.0),
            path_offset_mm=params.get(
                "path_offset_mm", params.get("offset_mm", 0.0)
            ),
            cut_side=cut_side,
        )
