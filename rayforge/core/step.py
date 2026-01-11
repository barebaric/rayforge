from __future__ import annotations
import logging
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Dict, Any, cast, Set
from blinker import Signal

from .item import DocItem
from .matrix import Matrix
from .capability import Capability, CAPABILITIES_BY_NAME

if TYPE_CHECKING:
    from .layer import Layer
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
        self.generated_workpiece_uid: Optional[str] = None
        self.applied_recipe_uid: Optional[str] = None
        self.capabilities: Set[Capability] = set()

        # Configuration for the pipeline, stored as dictionaries.
        self.modifiers_dicts: List[Dict[str, Any]] = []
        self.opsproducer_dict: Optional[Dict[str, Any]] = None
        self.per_workpiece_transformers_dicts: List[Dict[str, Any]] = []
        self.per_step_transformers_dicts: List[Dict[str, Any]] = []

        self.pixels_per_mm = 50, 50

        # Signals for notifying of model changes
        self.per_step_transformer_changed = Signal()
        self.visibility_changed = Signal()

        # Default machine-dependent values.
        self.power = 1.0
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
            "generated_workpiece_uid": self.generated_workpiece_uid,
            "applied_recipe_uid": self.applied_recipe_uid,
            "capabilities": [c.name for c in self.capabilities],
            "modifiers_dicts": self.modifiers_dicts,
            "opsproducer_dict": self.opsproducer_dict,
            "per_workpiece_transformers_dicts": (
                self.per_workpiece_transformers_dicts
            ),
            "per_step_transformers_dicts": self.per_step_transformers_dicts,
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
        step.generated_workpiece_uid = data.get("generated_workpiece_uid")
        step.applied_recipe_uid = data.get("applied_recipe_uid")

        # Deserialize capabilities from list of names to instances
        cap_names = data.get("capabilities", [])
        step.capabilities = {
            CAPABILITIES_BY_NAME[name]
            for name in cap_names
            if name in CAPABILITIES_BY_NAME
        }

        step.modifiers_dicts = data["modifiers_dicts"]
        step.opsproducer_dict = data["opsproducer_dict"]
        step.per_workpiece_transformers_dicts = data[
            "per_workpiece_transformers_dicts"
        ]
        step.per_step_transformers_dicts = data["per_step_transformers_dicts"]
        step.pixels_per_mm = data.get("pixels_per_mm", (50, 50))
        step.power = data.get("power", 1.0)
        step.max_power = data.get("max_power", 1000)
        step.cut_speed = data.get("cut_speed", 500)
        step.max_cut_speed = data.get("max_cut_speed", 10000)
        step.travel_speed = data.get("travel_speed", 5000)
        step.max_travel_speed = data.get("max_travel_speed", 10000)
        step.air_assist = data.get("air_assist", False)
        step.kerf_mm = data.get("kerf_mm", 0.0)
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
            "generated_workpiece_uid": self.generated_workpiece_uid,
        }

    @property
    def layer(self) -> Optional["Layer"]:
        """Returns the parent layer, if it exists."""
        # Local import to prevent circular dependency at module load time
        from .layer import Layer

        workflow = self.workflow
        if not workflow:
            return None

        layer = workflow.parent
        return layer if isinstance(layer, Layer) else None

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

    def set_power(self, power: float):
        if not (0.0 <= power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
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
        power_percent = int(self.power * 100)
        speed = int(self.cut_speed)
        return f"{power_percent}% power, {speed} mm/min"

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)
