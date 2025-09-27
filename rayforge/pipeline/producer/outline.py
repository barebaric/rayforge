from typing import List, Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...image.tracing import trace_surface
from ...core.geo import Geometry, contours
from ...core.matrix import Matrix
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class OutlineTracer(OpsProducer):
    """
    Uses a tracer and filters the results to keep only the
    outermost paths of a shape, ignoring any holes.
    """

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError("OutlineTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )

        source_contours: List[Geometry] = []
        outline_ops = Ops()
        coord_system = CoordinateSystem.MILLIMETER_SPACE
        source_dims = workpiece.size

        # If the workpiece has vectors, process them.
        if (
            workpiece
            and workpiece.vectors
            and not workpiece.vectors.is_empty()
        ):
            source_contours = workpiece.vectors.split_into_contours()

            external_contours = contours.filter_to_external_contours(
                source_contours
            )
            for geo in external_contours:
                outline_ops.extend(Ops.from_geometry(geo))

            # Apply the physical scale from the workpiece's matrix.
            sx, sy = workpiece.matrix.get_abs_scale()
            scaling_matrix = Matrix.scale(sx, sy)
            outline_ops.transform(scaling_matrix.to_4x4_numpy())

        # If no vectors, fall back to raster tracing the surface.
        else:
            source_contours = trace_surface(surface)

            external_contours = contours.filter_to_external_contours(
                source_contours
            )
            for geo in external_contours:
                outline_ops.extend(Ops.from_geometry(geo))

            # Scale the pixel-based ops to their final millimeter size.
            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()

            if px_width > 0 and px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height
                scaling_matrix = Matrix.scale(scale_x, scale_y)
                outline_ops.transform(scaling_matrix.to_4x4_numpy())

        final_ops.extend(outline_ops)
        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        return PipelineArtifact(
            ops=final_ops,
            is_scalable=True,
            source_coordinate_system=coord_system,
            source_dimensions=source_dims,
            generation_size=source_dims,
        )
