from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Dict, Any, Optional

from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from ...core.ops import Ops
from .base import OpsProducer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.progress import ProgressContext


class PlaceholderProducer(OpsProducer):
    """
    Producer that preserves configuration for unknown producer types.

    This is used when a document contains a step whose producer type is
    not available (e.g., because the addon that provides it is not installed).
    The placeholder preserves the original configuration so the document can
    be saved without data loss.
    """

    label = _("Missing Producer")

    def __init__(self, original_type: str, config: dict):
        self._original_type = original_type
        self._config = config.copy()
        self.label = _("Missing: {}").format(original_type)

    @property
    def original_type(self) -> str:
        """Returns the original producer type name that was not found."""
        return self._original_type

    def to_dict(self) -> dict:
        """Returns the preserved original configuration."""
        return self._config.copy()

    @property
    def supports_kerf(self) -> bool:
        return False

    @property
    def supports_cut_speed(self) -> bool:
        return False

    @property
    def supports_power(self) -> bool:
        return False

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional["ProgressContext"] = None,
    ) -> WorkPieceArtifact:
        size = workpiece.size if workpiece else (0.0, 0.0)
        return WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=size,
            generation_size=size,
            generation_id=generation_id,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "PlaceholderProducer":
        original_type = data.get("type", "Unknown")
        return cls(original_type, data)
