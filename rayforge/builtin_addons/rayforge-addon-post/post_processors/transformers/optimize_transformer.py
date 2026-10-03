import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.ops.transform.merge_scanlines import MergeScanlinesSpec
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
    1. Acceleration-aware scanline merging (when enabled): parallel
       cut lines on the same scan row, including rows spanning
       multiple workpieces, are bridged at zero power when the
       machine's motion profile makes the merged sweep faster.
    2. Workpiece-level reordering (when multiple workpieces are
       present).
    3. Segment-level k-d tree nearest-neighbor + 2-opt refinement.
    """

    SPEC_NAME = "optimize"

    DEFAULT_MERGE_TOLERANCE = 0.05
    DEFAULT_MERGE_MAX_GAP = 0.0

    def __init__(
        self,
        enabled: bool = True,
        allow_flip: bool = True,
        preserve_first: bool = False,
        preserve_order: list[str] | None = None,
        merge_scanlines: bool = True,
        merge_max_gap_mm: float = DEFAULT_MERGE_MAX_GAP,
        merge_tolerance: float = DEFAULT_MERGE_TOLERANCE,
        **kwargs,
    ):
        super().__init__(enabled=enabled, **kwargs)
        self.allow_flip = allow_flip
        self.preserve_first = preserve_first
        self.preserve_order = preserve_order or []
        self.merge_scanlines = merge_scanlines
        self.merge_max_gap_mm = merge_max_gap_mm
        self.merge_tolerance = merge_tolerance

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
        settings = settings or {}
        merge = None
        if self.merge_scanlines:
            merge = MergeScanlinesSpec(
                acceleration=float(
                    settings.get("machine_acceleration", 1000.0)
                ),
                cut_speed=float(settings.get("machine_max_cut_speed", 1000.0)),
                rapid_speed=float(
                    settings.get("machine_max_travel_speed", 1000.0)
                ),
                max_gap_mm=self.merge_max_gap_mm,
                tolerance=self.merge_tolerance,
            )
        return OptimizeSpec(
            allow_flip=self.allow_flip,
            preserve_first=self.preserve_first,
            preserve_order=list(self.preserve_order),
            merge_scanlines=merge,
        )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["allow_flip"] = self.allow_flip
        result["preserve_first"] = self.preserve_first
        result["preserve_order"] = self.preserve_order
        result["merge_scanlines"] = self.merge_scanlines
        result["merge_max_gap_mm"] = self.merge_max_gap_mm
        result["merge_tolerance"] = self.merge_tolerance
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
            merge_scanlines=data.get("merge_scanlines", True),
            merge_max_gap_mm=data.get(
                "merge_max_gap_mm", cls.DEFAULT_MERGE_MAX_GAP
            ),
            merge_tolerance=data.get(
                "merge_tolerance", cls.DEFAULT_MERGE_TOLERANCE
            ),
        )
