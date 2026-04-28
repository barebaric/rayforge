from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, TYPE_CHECKING, cast
from gettext import gettext as _

from rayforge.core.capability import Capability, MATERIAL_TEST
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
        assert OverscanTransformer is not None
        assert Optimize is not None
        optimize_dict = Optimize().to_dict()
        return [
            OverscanTransformer(
                enabled=True, distance_mm=0, auto=True
            ).to_dict(),
            optimize_dict,
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
        auto_distance = OverscanTransformer.calculate_auto_distance(
            step.cut_speed, machine.acceleration
        )
        for t in per_wp:
            if t.get("name") == "OverscanTransformer":
                t["distance_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = machine.get_default_head().uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
