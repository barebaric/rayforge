from typing import Optional, List, Dict, Any, TYPE_CHECKING
from gettext import gettext as _

from rayforge.core.geo import Point3D
from rayforge.core.geo.smooth import (
    compute_gaussian_kernel,
    smooth_polyline,
)
from rayforge.core.ops import Ops, LineToCommand, MoveToCommand
from rayforge.core.workpiece import WorkPiece
from rayforge.shared.tasker.progress import ProgressContext
from rayforge.pipeline.transformer.base import OpsTransformer, ExecutionPhase

if TYPE_CHECKING:
    from rayforge.core.geo import Geometry


class Smooth(OpsTransformer):
    """Smooths path segments using a Gaussian filter.

    This transformer uses a multi-stage "divide and conquer" algorithm:

    1.  **Dynamic Subdivision:** It resamples the path into a high-density
        set of points. The density is proportional to the smoothing
        'amount', ensuring perfect curves even for tiny radii.
    2.  **Anchor Detection:** It identifies all "anchor" points—endpoints
        and sharp corners—that must be preserved.
    3.  **Split & Smooth:** The path is split into independent sub-segments
        between these anchors. Each sub-segment is smoothed in isolation,
        preventing smoothing from "bleeding" across sharp corners.
    4.  **Reassembly:** The smoothed sub-segments are reassembled into the
        final, high-quality path.
    """

    def __init__(
        self, enabled: bool = True, amount=20, corner_angle_threshold=45
    ):
        """Initializes the smoothing filter.

        Args:
            enabled: Whether the transformer is active.
            amount: The smoothing strength (0-100) controlling the curve
                    radius.
            corner_angle_threshold: Corners with an internal angle (in
                                    degrees) smaller than this are
                                    preserved.
        """
        super().__init__(enabled=enabled)
        self._corner_angle_threshold = corner_angle_threshold
        self._kernel: Optional[List[float]] = None
        self._sigma: float = 0.1
        self._amount = -1
        self.amount = amount

    @property
    def execution_phase(self) -> ExecutionPhase:
        """Smooth needs to run on continuous paths before they are broken."""
        return ExecutionPhase.GEOMETRY_REFINEMENT

    @property
    def amount(self) -> int:
        """The smoothing strength, from 0 (none) to 100 (heavy)."""
        return self._amount

    @amount.setter
    def amount(self, value: int) -> None:
        """Updates the smoothing amount and pre-computes the kernel."""
        new_amount = max(0, min(100, value))
        if self._amount == new_amount:
            return
        self._amount = new_amount
        self._kernel, self._sigma = compute_gaussian_kernel(new_amount)
        self.changed.send(self)

    @property
    def corner_angle_threshold(self) -> float:
        """The corner angle threshold in degrees."""
        return self._corner_angle_threshold

    @corner_angle_threshold.setter
    def corner_angle_threshold(self, value_deg: float):
        """Sets the corner angle threshold from a value in degrees."""
        if self._corner_angle_threshold == value_deg:
            return
        self._corner_angle_threshold = value_deg
        self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Smooth Path")

    @property
    def description(self) -> str:
        return _("Smooths the path by applying a Gaussian filter.")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Executes the smoothing transformation on a set of operations.

        Args:
            ops: The operations object containing path data.
            context: Optional progress context for reporting progress.
        """
        if self.amount == 0:
            return

        ops.linearize_all()
        segments = list(ops.segments())
        ops.clear()
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            if context and context.is_cancelled():
                break
            points_to_smooth: Optional[List[Point3D]] = None
            if self._is_line_only_segment(segment):
                # Extract points. The `end` property may be typed as Optional.
                points_to_smooth = [
                    cmd.end for cmd in segment if cmd.end is not None
                ]
                smoothed = smooth_polyline(
                    points_to_smooth,
                    self.amount,
                    self.corner_angle_threshold,
                )
                if smoothed:
                    ops.move_to(*smoothed[0])
                    for point in smoothed[1:]:
                        ops.line_to(*point)
            else:
                # For complex segments (e.g., with curves) or malformed ones,
                # add them back without modification.
                for command in segment:
                    ops.add(command)

            if context and total_segments > 0:
                context.set_progress((i + 1) / total_segments)

    def _is_line_only_segment(self, segment: List) -> bool:
        """Checks if a segment contains only MoveTo and LineTo commands."""
        return (
            len(segment) > 1
            and isinstance(segment[0], MoveToCommand)
            and all(isinstance(c, LineToCommand) for c in segment[1:])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "amount": self.amount,
                "corner_angle_threshold": self.corner_angle_threshold,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Smooth":
        """Creates a Smooth instance from a dictionary."""
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            amount=data.get("amount", 20),
            corner_angle_threshold=data.get("corner_angle_threshold", 45),
        )
