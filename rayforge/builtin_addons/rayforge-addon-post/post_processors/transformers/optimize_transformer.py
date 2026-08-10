import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.ops.transform.optimize import OptimizeSpec

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import OpsTransformer

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

    SPEC_NAME = "optimize"

    def __init__(
        self,
        enabled: bool = True,
        allow_flip: bool = True,
        preserve_first: bool = False,
        preserve_order: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(enabled=enabled, **kwargs)
        self.allow_flip = allow_flip
        self.preserve_first = preserve_first
        self.preserve_order = preserve_order or []

    @property
    def label(self) -> str:
        return _("Optimize Path")

    @property
    def description(self) -> str:
        return _("Minimizes travel distance by reordering segments.")

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: list["Geometry"] | None,
        settings: dict[str, Any] | None,
    ) -> OptimizeSpec:
        return OptimizeSpec(
            allow_flip=self.allow_flip,
            preserve_first=self.preserve_first,
            preserve_order=list(self.preserve_order),
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["allow_flip"] = self.allow_flip
        result["preserve_first"] = self.preserve_first
        result["preserve_order"] = self.preserve_order
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Optimize":
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
