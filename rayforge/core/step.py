from __future__ import annotations
import logging
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Dict, Any, cast
from blinker import Signal

from .item import DocItem

if TYPE_CHECKING:
    from .workflow import Workflow


logger = logging.getLogger(__name__)


class Step(DocItem, ABC):
    """
    A set of modifiers and an OpsProducer that operate on WorkPieces.

    A Step is a stateless configuration object that defines a single
    operation (e.g., outline, engrave) to be performed. It holds its
    configuration as serializable dictionaries.
    """

    def __init__(
        self,
        typelabel: str,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or typelabel)
        self.typelabel = typelabel
        self.visible = True

        # Configuration for the pipeline, stored as dictionaries.
        # - ops-transformers are used per single workpiece.
        # - post-step transformers are applied on the combined ops
        #   of all workpieces of this step.
        self.modifiers_dicts: List[Dict[str, Any]] = []
        self.opsproducer_dict: Optional[Dict[str, Any]] = None
        self.opstransformers_dicts: List[Dict[str, Any]] = []
        self.post_step_transformers_dicts: List[Dict[str, Any]] = []
        self.laser_dict: Optional[Dict[str, Any]] = None

        self.pixels_per_mm = 50, 50

        # Signals for notifying of model changes
        self.post_step_transformer_changed = Signal()
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

    def to_dict(self) -> Dict:
        """Serializes the step and its configuration to a dictionary."""
        return {
            "uid": self.uid,
            "type": "step",
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "typelabel": self.typelabel,
            "visible": self.visible,
            "modifiers_dicts": self.modifiers_dicts,
            "opsproducer_dict": self.opsproducer_dict,
            "opstransformers_dicts": self.opstransformers_dicts,
            "post_step_transformers_dicts": self.post_step_transformers_dicts,
            "laser_dict": self.laser_dict,
            "pixels_per_mm": self.pixels_per_mm,
            "power": self.power,
            "max_power": self.max_power,
            "cut_speed": self.cut_speed,
            "max_cut_speed": self.max_cut_speed,
            "travel_speed": self.travel_speed,
            "max_travel_speed": self.max_travel_speed,
            "air_assist": self.air_assist,
            "children": [child.to_dict() for child in self.children],
        }

    @property
    def workflow(self) -> Optional["Workflow"]:
        """Returns the parent workflow, if it exists."""
        # Local import to prevent circular dependency at module load time
        from .workflow import Workflow

        if self.parent and isinstance(self.parent, Workflow):
            return cast(Workflow, self.parent)
        return None

    def set_visible(self, visible: bool):
        self.visible = visible
        self.visibility_changed.send(self)

    def set_power(self, power: int):
        self.power = power
        self.updated.send(self)

    def set_cut_speed(self, speed: int):
        self.cut_speed = int(speed)
        self.updated.send(self)

    def set_travel_speed(self, speed: int):
        self.travel_speed = int(speed)
        self.updated.send(self)

    def set_air_assist(self, enabled: bool):
        self.air_assist = bool(enabled)
        self.updated.send(self)

    def get_summary(self) -> str:
        power_percent = (
            int(self.power / self.max_power * 100) if self.max_power else 0
        )
        speed = int(self.cut_speed)
        return f"{power_percent}% power, {speed} mm/min"

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)
