from __future__ import annotations
import logging
import uuid
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from blinker import Signal

if TYPE_CHECKING:
    from .doc import Doc
    from .workflow import Workflow


logger = logging.getLogger(__name__)


class Step(ABC):
    """
    A set of modifiers and an OpsProducer that operate on WorkPieces.

    A Step is a stateless configuration object that defines a single
    operation (e.g., outline, engrave) to be performed. It holds its
    configuration as serializable dictionaries.
    """

    def __init__(
        self,
        workflow: "Workflow",
        typelabel: str,
        name: Optional[str] = None,
    ):
        self.workflow: Optional["Workflow"] = workflow
        self.uid = str(uuid.uuid4())
        self.typelabel = typelabel
        self.name = name or self.typelabel
        self.visible = True

        # Configuration for the pipeline, stored as dictionaries
        self.modifiers_dicts: List[Dict[str, Any]] = []
        self.opsproducer_dict: Optional[Dict[str, Any]] = None
        self.opstransformers_dicts: List[Dict[str, Any]] = []
        self.laser_dict: Optional[Dict[str, Any]] = None

        self.passes: int = 1
        self.pixels_per_mm = 50, 50

        # Signals for notifying of model changes
        self.changed = Signal()
        self.visibility_changed = Signal()

        # Default machine-dependent values. These will be overwritten by
        # the step factories in the pipeline module.
        self.power = 1000
        self.max_power = 1000
        self.cut_speed = 500
        self.max_cut_speed = 10000
        self.travel_speed = 5000
        self.max_travel_speed = 10000
        self.air_assist = False

    @property
    def doc(self) -> Optional["Doc"]:
        """The parent Doc object, accessed via the Workflow."""
        return self.workflow.doc if self.workflow else None

    def set_passes(self, passes: int):
        self.passes = int(passes)
        self.changed.send(self)

    def set_visible(self, visible: bool):
        self.visible = visible
        self.visibility_changed.send(self)

    def set_power(self, power: int):
        self.power = power
        self.changed.send(self)

    def set_cut_speed(self, speed: int):
        self.cut_speed = int(speed)
        self.changed.send(self)

    def set_travel_speed(self, speed: int):
        self.travel_speed = int(speed)
        self.changed.send(self)

    def set_air_assist(self, enabled: bool):
        self.air_assist = bool(enabled)
        self.changed.send(self)

    def get_summary(self) -> str:
        power_percent = (
            int(self.power / self.max_power * 100) if self.max_power else 0
        )
        speed = int(self.cut_speed)
        return f"{power_percent}% power, {speed} mm/min"

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)
