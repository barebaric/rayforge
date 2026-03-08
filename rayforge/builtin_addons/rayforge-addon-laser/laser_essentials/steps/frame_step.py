from __future__ import annotations

from typing import Optional, Set, TYPE_CHECKING
from gettext import gettext as _

from rayforge.core.capability import CUT, SCORE, Capability
from rayforge.core.step import Step
from rayforge.pipeline.transformer import (
    CropTransformer,
    MultiPassTransformer,
    Optimize,
    TabOpsTransformer,
)
from ..producers import FrameProducer


if TYPE_CHECKING:
    from rayforge.context import RayforgeContext


class FrameStep(Step):
    TYPELABEL = _("Frame Outline")
    DEFAULT_CAPABILITIES: Set[Capability] = {CUT, SCORE}
    PRODUCER_CLASS = FrameProducer

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.capabilities = self.DEFAULT_CAPABILITIES.copy()

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        return [
            TabOpsTransformer().to_dict(),
            CropTransformer(enabled=False).to_dict(),
            Optimize().to_dict(),
        ], [
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "FrameStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        per_wp, per_step = cls.get_default_transformers_dicts()
        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
