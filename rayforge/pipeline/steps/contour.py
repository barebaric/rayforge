from __future__ import annotations

from typing import Optional, Set, TYPE_CHECKING
from gettext import gettext as _

from ...core.capability import CUT, SCORE, Capability
from ...core.step import Step
from ..producer import ContourProducer
from ..transformer import (
    MultiPassTransformer,
    Optimize,
    Smooth,
    TabOpsTransformer,
)

if TYPE_CHECKING:
    from ...context import RayforgeContext


class ContourStep(Step):
    TYPELABEL = _("Contour")
    DEFAULT_CAPABILITIES: Set[Capability] = {CUT, SCORE}
    PRODUCER_CLASS = ContourProducer

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
        optimize: bool = True,
    ) -> "ContourStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        step.per_workpiece_transformers_dicts = [
            Smooth(enabled=False, amount=20).to_dict(),
            TabOpsTransformer().to_dict(),
        ]
        if optimize:
            step.per_workpiece_transformers_dicts.append(
                Optimize().to_dict(),
            )
        step.per_step_transformers_dicts = [
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
