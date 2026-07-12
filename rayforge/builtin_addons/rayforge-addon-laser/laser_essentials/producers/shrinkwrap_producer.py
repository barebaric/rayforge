from typing import TYPE_CHECKING, Any, Dict, Optional

import cairo
import numpy as np
from raygeo.ops import Ops
from raygeo.ops.assembly.shrinkwrap import shrinkwrap

from rayforge.image.tracing import prepare_surface
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.producer.base import CutSide, OpsProducer
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector,
    make_artifact,
    wrap_assembler_result,
)
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser


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
        cut_side: CutSide = CutSide.CENTERLINE,
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
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError(
                "ShrinkWrapProducer requires a workpiece context."
            )

        settings = settings or {}
        kerf_mm = settings.get("kerf_mm", laser.spot_size_mm[0])

        # 1. Prepare boolean image (Cairo-specific)
        boolean_image = prepare_surface(surface)

        if not np.any(boolean_image):
            return make_artifact(
                Ops(), workpiece, generation_id, is_vector=True
            )

        part = build_part_vector(workpiece)
        if part is None:
            raise ValueError(
                "ShrinkWrapProducer: workpiece has no vector geometry"
            )

        tolerance = settings.get("arc_tolerance", 0.03)
        allow_arcs = settings.get(
            "machine_supports_arcs",
            settings.get("output_arcs", True),
        )
        supports_curves = settings.get("machine_supports_curves", False)

        # 2. The shrinkwrap assembler computes the total offset
        #    from kerf, path offset, and cut side internally.
        result = shrinkwrap(
            part,
            boolean_image,
            gravity=self.gravity,
            kerf_mm=kerf_mm,
            path_offset_mm=self.path_offset_mm,
            cut_side=self.cut_side.name.lower(),
            arc_tolerance=tolerance,
            allow_arcs=allow_arcs,
            supports_curves=supports_curves,
        )

        return wrap_assembler_result(
            result,
            workpiece,
            laser,
            generation_id,
            set_power=settings.get("power", 0),
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
            "cut_side", params.get("kerf_mode", "CENTERLINE")
        )
        try:
            cut_side = CutSide[cut_side_str]
        except KeyError:
            cut_side = CutSide.CENTERLINE

        return cls(
            gravity=params.get("gravity", 0.0),
            path_offset_mm=params.get(
                "path_offset_mm", params.get("offset_mm", 0.0)
            ),
            cut_side=cut_side,
        )
