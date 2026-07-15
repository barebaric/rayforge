from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, cast

from rayforge.core.capability import MATERIAL_TEST, Capability
from rayforge.core.step import Step
from rayforge.pipeline.transformer.registry import transformer_registry

from ..producers import MaterialTestGridProducer

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class MaterialTestStep(Step):
    TYPELABEL = _("Material Test Grid")
    ICON = "test-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (MATERIAL_TEST,)
    PRODUCER_CLASS = MaterialTestGridProducer
    HIDDEN = True

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        OverscanTransformer = transformer_registry.get("OverscanTransformer")
        Optimize = transformer_registry.get("Optimize")
        BidirScanOffsetTransformer = transformer_registry.get(
            "BidirScanOffsetTransformer"
        )
        assert OverscanTransformer is not None
        assert Optimize is not None
        assert BidirScanOffsetTransformer is not None
        # Off by default: Optimize's nearest-neighbor travel reordering has
        # no concept of "cell" boundaries, so it can interleave lines from
        # different cells instead of engraving each one fully before moving
        # to the next. Left toggleable (rather than removed outright) so
        # it's easy to compare with/without.
        optimize_dict = Optimize(enabled=False).to_dict()
        return [
            OverscanTransformer(
                enabled=True, distance_mm=0, auto=True
            ).to_dict(),
            optimize_dict,
            BidirScanOffsetTransformer(enabled=True).to_dict(),
        ], [
            optimize_dict,
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "MaterialTestStep":
        machine = context.machine
        assert machine is not None

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        per_wp, per_step = cls.get_default_transformers_dicts()

        OverscanTransformer = cast(
            "OverscanTransformerType",
            transformer_registry.get("OverscanTransformer"),
        )
        assert OverscanTransformer is not None
        # Double the usual auto-calculated distance: individual test blocks
        # benefit from extra run-up/run-out so backlash settling happens
        # outside the visible engrave area, keeping it distinguishable from
        # whatever parameter the grid is testing.
        auto_distance = (
            OverscanTransformer.calculate_auto_distance(
                step.cut_speed, machine.acceleration
            )
            * 2
        )
        for t in per_wp:
            if t.get("name") == "OverscanTransformer":
                t["distance_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        default_head = machine.get_default_head()
        step.selected_laser_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
