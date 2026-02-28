from __future__ import annotations

from typing import Optional, Set, TYPE_CHECKING
from gettext import gettext as _

from ...core.capability import Capability
from ...core.step import Step
from ..producer import MaterialTestGridProducer

if TYPE_CHECKING:
    from ...context import RayforgeContext


class MaterialTestStep(Step):
    TYPELABEL = _("Material Test Grid")
    DEFAULT_CAPABILITIES: Set[Capability] = set()
    PRODUCER_CLASS = MaterialTestGridProducer
    HIDDEN = True

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
        **kwargs,
    ) -> "MaterialTestStep":
        machine = context.machine
        assert machine is not None

        step = cls(name=name)
        step.opsproducer_dict = cls.PRODUCER_CLASS().to_dict()
        step.per_workpiece_transformers_dicts = []
        step.per_step_transformers_dicts = []
        step.selected_laser_uid = machine.get_default_head().uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        return step
