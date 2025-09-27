from typing import Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...image.tracing import trace_surface
from ...core.matrix import Matrix
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class EdgeTracer(OpsProducer):
    """
    Uses the tracer to find all paths in a shape, including
    both external outlines and internal holes.
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
            raise ValueError("EdgeTracer requires a workpiece context.")

        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )
        source_dims = None
        coord_system = CoordinateSystem.PIXEL_SPACE

        if workpiece and workpiece.vectors:
            vector_ops = Ops.from_geometry(workpiece.vectors)

            # The producer's job is ONLY to apply scale, not position/rotation.
            # We extract the scale from the workpiece's full matrix and create
            # a new matrix that ONLY contains that scale.
            sx, sy = workpiece.matrix.get_abs_scale()
            scaling_matrix = Matrix.scale(sx, sy)

            # Now we transform using the scale-only matrix, preserving your
            # correct use of .to_4x4_numpy().
            vector_ops.transform(scaling_matrix.to_4x4_numpy())

            final_ops.extend(vector_ops)
            coord_system = CoordinateSystem.MILLIMETER_SPACE
            source_dims = workpiece.size

        else:  # Fall back to raster tracing
            geometries = trace_surface(surface)

            raster_trace_ops = Ops()
            for geo in geometries:
                raster_trace_ops.extend(Ops.from_geometry(geo))

            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()

            if px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height
                scaling_matrix = Matrix.scale(scale_x, scale_y)
                raster_trace_ops.transform(scaling_matrix.to_4x4_numpy())

            final_ops.extend(raster_trace_ops)
            coord_system = CoordinateSystem.MILLIMETER_SPACE
            source_dims = workpiece.size

        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))
        return PipelineArtifact(
            ops=final_ops,
            is_scalable=True,
            source_coordinate_system=coord_system,
            source_dimensions=source_dims,
        )
