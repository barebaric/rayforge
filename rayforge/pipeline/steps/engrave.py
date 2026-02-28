from __future__ import annotations

from typing import Optional, Set, TYPE_CHECKING
from gettext import gettext as _

from ...core.capability import ENGRAVE, Capability
from ...core.step import Step
from ..producer import Rasterizer
from ..transformer import MultiPassTransformer, Optimize, OverscanTransformer

if TYPE_CHECKING:
    from ...context import RayforgeContext


class EngraveStep(Step):
    TYPELABEL = _("Engrave")
    DEFAULT_CAPABILITIES: Set[Capability] = {ENGRAVE}
    PRODUCER_CLASS = Rasterizer

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.capabilities = self.DEFAULT_CAPABILITIES.copy()

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
    ) -> "EngraveStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        auto_distance = OverscanTransformer.calculate_auto_distance(
            machine.max_cut_speed, machine.acceleration
        )

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        step.per_workpiece_transformers_dicts = [
            OverscanTransformer(
                enabled=True, distance_mm=auto_distance, auto=True
            ).to_dict(),
            Optimize().to_dict(),
        ]
        step.per_step_transformers_dicts = [
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]
        step.selected_laser_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
