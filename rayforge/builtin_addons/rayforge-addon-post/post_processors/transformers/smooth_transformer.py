from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops.transform.smooth import SmoothSpec

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry


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
        """Updates the smoothing amount."""
        new_amount = max(0, min(100, value))
        if self._amount == new_amount:
            return
        self._amount = new_amount
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

    def to_spec(
        self,
        workpiece: Optional[WorkPiece],
        stock_geometries: Optional[List["Geometry"]],
        settings: Optional[Dict[str, Any]],
    ) -> SmoothSpec:
        return SmoothSpec(
            amount=self.amount,
            corner_angle_threshold=self.corner_angle_threshold,
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
