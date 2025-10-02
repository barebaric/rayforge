from __future__ import annotations
import logging
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Dict, Any, cast
from blinker import Signal

from .item import DocItem
from .matrix import Matrix

if TYPE_CHECKING:
    from .workflow import Workflow
    from ..machine.models.machine import Machine
    from ..machine.models.laser import Laser


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
        self.selected_laser_uid: Optional[str] = None

        # Configuration for the pipeline, stored as dictionaries.
        # - ops-transformers are used per single workpiece.
        # - post-step transformers are applied on the combined ops
        #   of all workpieces of this step.
        self.modifiers_dicts: List[Dict[str, Any]] = []
        self.opsproducer_dict: Optional[Dict[str, Any]] = None
        self.opstransformers_dicts: List[Dict[str, Any]] = []
        self.post_step_transformers_dicts: List[Dict[str, Any]] = []

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
        self.kerf_mm: float = 0.0

    def to_dict(self) -> Dict:
        """Serializes the step and its configuration to a dictionary."""
        return {
            "uid": self.uid,
            "type": "step",
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "typelabel": self.typelabel,
            "visible": self.visible,
            "selected_laser_uid": self.selected_laser_uid,
            "modifiers_dicts": self.modifiers_dicts,
            "opsproducer_dict": self.opsproducer_dict,
            "opstransformers_dicts": self.opstransformers_dicts,
            "post_step_transformers_dicts": self.post_step_transformers_dicts,
            "pixels_per_mm": self.pixels_per_mm,
            "power": self.power,
            "max_power": self.max_power,
            "cut_speed": self.cut_speed,
            "max_cut_speed": self.max_cut_speed,
            "travel_speed": self.travel_speed,
            "max_travel_speed": self.max_travel_speed,
            "air_assist": self.air_assist,
            "kerf_mm": self.kerf_mm,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """Deserializes a Step instance from a dictionary."""
        step = cls(typelabel=data["typelabel"], name=data.get("name"))
        step.uid = data["uid"]
        step.matrix = Matrix.from_list(data["matrix"])
        step.visible = data["visible"]
        step.selected_laser_uid = data.get("selected_laser_uid")
        step.modifiers_dicts = data["modifiers_dicts"]
        step.opsproducer_dict = data["opsproducer_dict"]
        step.opstransformers_dicts = data["opstransformers_dicts"]
        step.post_step_transformers_dicts = data[
            "post_step_transformers_dicts"
        ]
        step.pixels_per_mm = data["pixels_per_mm"]
        step.power = data["power"]
        step.max_power = data["max_power"]
        step.cut_speed = data["cut_speed"]
        step.max_cut_speed = data["max_cut_speed"]
        step.travel_speed = data["travel_speed"]
        step.max_travel_speed = data["max_travel_speed"]
        step.air_assist = data["air_assist"]
        step.kerf_mm = data["kerf_mm"]
        return step

    def get_settings(self) -> Dict[str, Any]:
        """
        Bundles all physical process parameters into a dictionary.
        Only includes settings of the step itself, and not of producer,
        transformer, etc.
        """
        return {
            "power": self.power,
            "cut_speed": self.cut_speed,
            "travel_speed": self.travel_speed,
            "air_assist": self.air_assist,
            "pixels_per_mm": self.pixels_per_mm,
            "kerf_mm": self.kerf_mm,
        }

    @property
    def workflow(self) -> Optional["Workflow"]:
        """Returns the parent workflow, if it exists."""
        # Local import to prevent circular dependency at module load time
        from .workflow import Workflow

        if self.parent and isinstance(self.parent, Workflow):
            return cast(Workflow, self.parent)
        return None

    @property
    def show_general_settings(self) -> bool:
        """
        Returns whether general settings (power, speed, air assist) should be
        shown in the settings dialog. Override in subclasses to hide these
        settings when they don't apply.
        """
        return True

    def get_selected_laser(self, machine: "Machine") -> "Laser":
        """
        Resolves and returns the selected Laser instance for this step.
        Falls back to the first available laser on the machine if the
        selection is invalid or not set.
        """
        if self.selected_laser_uid:
            for head in machine.heads:
                if head.uid == self.selected_laser_uid:
                    return head
        # Fallback
        if not machine.heads:
            raise ValueError("Machine has no laser heads configured.")
        return machine.heads[0]

    def set_selected_laser_uid(self, uid: Optional[str]):
        """
        Sets the UID of the laser to be used by this step.
        """
        if self.selected_laser_uid != uid:
            self.selected_laser_uid = uid
            self.updated.send(self)

    def set_visible(self, visible: bool):
        self.visible = visible
        self.visibility_changed.send(self)
        self.updated.send(self)

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

    def set_kerf_mm(self, kerf: float):
        """Sets the kerf (beam width) in millimeters for this process."""
        self.kerf_mm = float(kerf)
        self.updated.send(self)

    def get_summary(self) -> str:
        power_percent = (
            int(self.power / self.max_power * 100) if self.max_power else 0
        )
        speed = int(self.cut_speed)
        return f"{power_percent}% power, {speed} mm/min"

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)
