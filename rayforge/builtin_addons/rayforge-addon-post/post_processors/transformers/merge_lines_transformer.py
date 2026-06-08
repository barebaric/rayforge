from __future__ import annotations

from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Sequence,
)

from raygeo.ops import Ops

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo import Geometry


class MergeLinesTransformer(OpsTransformer):
    """
    Merges overlapping/collinear line segments across all paths.

    This transformer detects line segments that are collinear and overlapping
    (typically from adjacent workpieces sharing an edge) and replaces the
    covered sub-segments with travel moves to avoid cutting the same line
    twice.

    The transformer should run before optimization and MultiPassTransformer.
    """

    DEFAULT_TOLERANCE = 0.01

    def __init__(
        self, enabled: bool = True, tolerance: float = DEFAULT_TOLERANCE
    ):
        super().__init__(enabled=enabled)
        self._tolerance = tolerance

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self._tolerance = max(0.001, value)
        self.changed.send(self)

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Merge Lines")

    @property
    def description(self) -> str:
        return _("Merges overlapping lines to avoid double passing.")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[Sequence["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        if ops.is_empty():
            return

        ops.merge_overlapping_lines(self._tolerance)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["tolerance"] = self._tolerance
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeLinesTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            tolerance=data.get("tolerance", cls.DEFAULT_TOLERANCE),
        )
