from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple

from rayforge.core.capability import CUT, Capability
from rayforge.core.step import Step

from ..producers import WavefrontProducer

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


class WavefrontStep(Step):
    TYPELABEL = _("Wavefront")
    ICON = "step-contour-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (CUT,)
    PRODUCER_CLASS = WavefrontProducer

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        return [], []

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "WavefrontStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        step.per_workpiece_transformers_dicts = []
        step.per_step_transformers_dicts = []
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
