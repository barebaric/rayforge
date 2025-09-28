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
from ...core.geo import contours

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class EdgeTracer(OpsProducer):
    """
    Uses the Potrace engine to trace paths in a shape. Can optionally trace
    only the outermost paths, ignoring any holes.
    """

    def __init__(self, remove_inner_paths: bool = False):
        """
        Initializes the EdgeTracer.

        Args:
            remove_inner_paths: If True, only the outermost paths (outlines)
                                are traced, and inner holes are ignored.
                                Defaults to False.
        """
        super().__init__()
        self.remove_inner_paths = remove_inner_paths

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

        processed_ops = Ops()
        is_vector_source = (
            workpiece
            and workpiece.vectors
            and not workpiece.vectors.is_empty()
        )

        if is_vector_source:
            if self.remove_inner_paths:
                source_contours = workpiece.vectors.split_into_contours()
                external_contours = contours.filter_to_external_contours(
                    source_contours
                )
                for geo in external_contours:
                    processed_ops.extend(Ops.from_geometry(geo))
            else:
                processed_ops = Ops.from_geometry(workpiece.vectors)

            # The producer's job is ONLY to apply scale, not position/rotation.
            sx, sy = workpiece.matrix.get_abs_scale()
            scaling_matrix = Matrix.scale(sx, sy)
            processed_ops.transform(scaling_matrix.to_4x4_numpy())

        else:  # Fall back to raster tracing
            source_contours = trace_surface(surface)

            if self.remove_inner_paths:
                target_contours = contours.filter_to_external_contours(
                    source_contours
                )
            else:
                target_contours = source_contours

            for geo in target_contours:
                processed_ops.extend(Ops.from_geometry(geo))

            # Scale the pixel-based ops to their final millimeter size.
            width_mm, height_mm = workpiece.size
            px_width, px_height = surface.get_width(), surface.get_height()

            if px_width > 0 and px_height > 0:
                scale_x = width_mm / px_width
                scale_y = height_mm / px_height
                scaling_matrix = Matrix.scale(scale_x, scale_y)
                processed_ops.transform(scaling_matrix.to_4x4_numpy())

        final_ops.extend(processed_ops)
        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        return PipelineArtifact(
            ops=final_ops,
            is_scalable=is_vector_source,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {"remove_inner_paths": self.remove_inner_paths},
        }
