import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops import Ops

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo.geo import Geometry


logger = logging.getLogger(__name__)


class Optimize(OpsTransformer):
    """
    Optimizes toolpaths to minimize travel distance.

    Delegates to the Rust-based ``Ops.optimize_travel()`` which performs:
    1. Workpiece-level reordering (when multiple workpieces are present).
    2. Segment-level k-d tree nearest-neighbor + 2-opt refinement.
    """

    def __init__(
        self,
        enabled: bool = True,
        allow_flip: bool = True,
        preserve_first: bool = False,
        preserve_order: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(enabled=enabled, **kwargs)
        self.allow_flip = allow_flip
        self.preserve_first = preserve_first
        self.preserve_order = preserve_order or []

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Optimize Path")

    @property
    def description(self) -> str:
        return _("Minimizes travel distance by reordering segments.")

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if context is None:
            return
        if context.is_cancelled():
            return

        context.set_total(1.0)

        class _Cb:
            def is_cancelled(self) -> bool:
                return context.is_cancelled()

            def __call__(self, progress: float, message: str) -> None:
                context.set_progress(progress)
                if message:
                    context.set_message(message)

        ops.optimize_travel(
            allow_flip=self.allow_flip,
            preserve_first=self.preserve_first,
            preserve_order=self.preserve_order,
            progress_cb=_Cb(),
        )

        logger.debug("Optimization finished")
        context.set_message(_("Optimization complete"))
        context.set_progress(1.0)
        context.flush()

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["allow_flip"] = self.allow_flip
        result["preserve_first"] = self.preserve_first
        result["preserve_order"] = self.preserve_order
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Optimize":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            allow_flip=data.get("allow_flip", True),
            preserve_first=data.get("preserve_first", False),
            preserve_order=data.get("preserve_order", []),
        )
