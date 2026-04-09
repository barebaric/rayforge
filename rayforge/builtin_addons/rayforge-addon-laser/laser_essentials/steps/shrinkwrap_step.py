from __future__ import annotations

from typing import Optional, Set, TYPE_CHECKING
from gettext import gettext as _

from rayforge.core.capability import CUT, SCORE, Capability
from rayforge.core.step import Step
from rayforge.pipeline.producer.base import CutSide
from rayforge.pipeline.transformer.registry import transformer_registry
from ..producers import ShrinkWrapProducer


if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


class ShrinkWrapStep(Step):
    TYPELABEL = _("Shrink Wrap")
    DEFAULT_CAPABILITIES: Set[Capability] = {CUT, SCORE}
    PRODUCER_CLASS = ShrinkWrapProducer

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.capabilities = self.DEFAULT_CAPABILITIES.copy()

    def get_operation_mode_short(self):
        if not self.opsproducer_dict:
            return None
        params = self.opsproducer_dict.get("params", {})
        cut_side_str = params.get("cut_side")
        if not cut_side_str:
            return None
        try:
            return CutSide[cut_side_str].label()
        except KeyError:
            return None

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        Smooth = transformer_registry.get("Smooth")
        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        TabOpsTransformer = transformer_registry.get("TabOpsTransformer")
        CropTransformer = transformer_registry.get("CropTransformer")
        MergeLinesTransformer = transformer_registry.get(
            "MergeLinesTransformer"
        )
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert Smooth is not None
        assert LeadInOutTransformer is not None
        assert TabOpsTransformer is not None
        assert CropTransformer is not None
        assert MergeLinesTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            Smooth(enabled=False, amount=20).to_dict(),
            LeadInOutTransformer(
                enabled=False, lead_in_mm=0, lead_out_mm=0, auto=True
            ).to_dict(),
            TabOpsTransformer().to_dict(),
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            MergeLinesTransformer().to_dict(),
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "ShrinkWrapStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        per_wp, per_step = cls.get_default_transformers_dicts()

        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(machine.max_cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
