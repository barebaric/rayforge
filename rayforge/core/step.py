from __future__ import annotations
import logging
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Dict, Any, cast, Tuple
from gettext import gettext as _
from blinker import Signal
from ..pipeline.transformer.registry import transformer_registry
from ..shared.units.formatter import format_value
from .capability import Capability
from .item import DocItem
from .matrix import Matrix
from .step_registry import step_registry

if TYPE_CHECKING:
    from ..context import RayforgeContext
    from ..machine.models.laser import Laser
    from ..machine.models.machine import Machine
    from .layer import Layer
    from .workflow import Workflow


logger = logging.getLogger(__name__)


class Step(DocItem, ABC):
    """
    An OpsProducer configuration that operates on WorkPieces.

    A Step is a stateless configuration object that defines a single
    operation (e.g., outline, engrave) to be performed. It holds its
    configuration as serializable dictionaries.
    """

    HIDDEN: bool = False
    ICON: str = ""
    CAPABILITIES: Tuple[Capability, ...] = ()

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

        self.opsproducer_dict: Optional[Dict[str, Any]] = None
        self.per_workpiece_transformers_dicts: List[Dict[str, Any]] = []
        self.per_step_transformers_dicts: List[Dict[str, Any]] = []

        self.pixels_per_mm = 50, 50

        # Signals for notifying of model changes
        self.per_step_transformer_changed = Signal()
        self.visibility_changed = Signal()

        # Default machine-dependent values.
        self.power: float = 1.0
        self.max_power = 1000
        self.cut_speed: int = 500
        self.max_cut_speed = 10000
        self.travel_speed: int = 5000
        self.max_travel_speed = 10000
        self.air_assist: bool = False
        self.kerf_mm: float = 0.0
        self.tab_power: float = 0.0

        # Forward compatibility: store unknown attributes
        self.extra: Dict[str, Any] = {}

        self._apply_capability_defaults()

    @property
    def capabilities(self) -> Tuple[Capability, ...]:
        return type(self).CAPABILITIES

    def _apply_capability_defaults(self):
        """
        Overwrites instance attributes with capability VarSet defaults.

        Iterates all capabilities defined on this step's class. For each
        Var in each capability's VarSet, validates that the attribute
        exists on the instance, then overwrites it with the Var's
        default. Raises AttributeError if a capability defines a key
        that does not exist as an instance attribute.
        """
        for cap in self.capabilities:
            for var in cap.varset:
                if not hasattr(self, var.key):
                    raise AttributeError(
                        f"{type(self).__name__} has no attribute "
                        f"'{var.key}' required by capability "
                        f"'{cap.name}'"
                    )
                setattr(self, var.key, var.default)

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "Step":
        """
        Factory method to create a fully configured step instance.

        Subclasses must override this to provide default configuration
        based on the context (e.g., machine settings).
        """
        raise NotImplementedError(
            f"{cls.__name__}.create() must be implemented by subclass"
        )

    def to_dict(self) -> Dict:
        """Serializes the step and its configuration to a dictionary."""
        result = {
            "uid": self.uid,
            "type": "step",
            "step_type": self.__class__.__name__,
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "typelabel": self.typelabel,
            "visible": self.visible,
            "selected_laser_uid": self.selected_laser_uid,
            "generated_workpiece_uid": self.generated_workpiece_uid,
            "applied_recipe_uid": self.applied_recipe_uid,
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
            "tab_power": self.tab_power,
            "children": [child.to_dict() for child in self.children],
        }
        result.update(self.extra)
        return result

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        """
        Returns default transformer configurations for this step type.

        Returns:
            A tuple of (per_workpiece_transformers_dicts,
            per_step_transformers_dicts) for new steps of this type.
            Subclasses should override this to provide their defaults.
        """
        return [], []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """Deserializes a Step instance from a dictionary."""
        known_keys = {
            "uid",
            "type",
            "step_type",
            "name",
            "matrix",
            "typelabel",
            "visible",
            "selected_laser_uid",
            "generated_workpiece_uid",
            "applied_recipe_uid",
            "modifiers_dicts",
            "opsproducer_dict",
            "per_workpiece_transformers_dicts",
            "per_step_transformers_dicts",
            "pixels_per_mm",
            "power",
            "max_power",
            "cut_speed",
            "max_cut_speed",
            "travel_speed",
            "max_travel_speed",
            "air_assist",
            "kerf_mm",
            "tab_power",
            "children",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        step_type_name = data.get("step_type")
        if step_type_name:
            step_class = step_registry.get(step_type_name)
        else:
            step_class = None

        if step_class is None:
            typelabel = data.get("typelabel")
            if typelabel:
                step_class = step_registry.get_by_typelabel(typelabel)

        if step_class is None:
            step_class = cls

        step = step_class(typelabel=data["typelabel"], name=data.get("name"))
        step.uid = data["uid"]
        step.matrix = Matrix.from_list(data["matrix"])
        step.visible = data["visible"]
        step.selected_laser_uid = data.get("selected_laser_uid")
        step.generated_workpiece_uid = data.get("generated_workpiece_uid")
        step.applied_recipe_uid = data.get("applied_recipe_uid")

        step.opsproducer_dict = data["opsproducer_dict"]
        loaded_per_wp = data["per_workpiece_transformers_dicts"]
        loaded_per_step = data["per_step_transformers_dicts"]

        default_per_wp, default_per_step = (
            step_class.get_default_transformers_dicts()
        )
        step.per_workpiece_transformers_dicts = Step._merge_transformer_dicts(
            loaded_per_wp, default_per_wp
        )
        step.per_step_transformers_dicts = Step._merge_transformer_dicts(
            loaded_per_step, default_per_step
        )

        # Share dict references for transformers that appear in both lists
        step._unify_shared_transformers()

        step.pixels_per_mm = data.get("pixels_per_mm", (100, 100))
        step.max_power = data.get("max_power", step.max_power)
        step.max_cut_speed = data.get("max_cut_speed", step.max_cut_speed)
        step.max_travel_speed = data.get(
            "max_travel_speed", step.max_travel_speed
        )
        step.power = data.get("power", step.power)
        step.cut_speed = data.get("cut_speed", step.cut_speed)
        step.travel_speed = data.get("travel_speed", step.travel_speed)
        step.air_assist = data.get("air_assist", step.air_assist)
        step.kerf_mm = data.get("kerf_mm", step.kerf_mm)
        step.tab_power = data.get("tab_power", step.tab_power)
        step.extra = extra
        return step

    @staticmethod
    def _merge_transformer_dicts(
        loaded: List[Dict[str, Any]], defaults: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merges loaded transformer dicts with defaults.

        Adds any transformers from defaults that are not present in loaded,
        preserving order where new transformers appear in the defaults list.
        """
        loaded_names = {t.get("name") for t in loaded if t.get("name")}
        result = list(loaded)
        for default_t in defaults:
            name = default_t.get("name")
            if name and name not in loaded_names:
                result.append(default_t)
        return result

    def _unify_shared_transformers(self):
        """
        Ensures transformers that appear in both lists share the same dict.

        Some transformers (like Optimize) are intended to be the same instance
        in both per_workpiece and per_step lists. This method detects such
        cases and unifies them to share the same dict reference.
        """
        per_wp_names = {
            t.get("name"): t
            for t in self.per_workpiece_transformers_dicts
            if t.get("name")
        }
        for i, t in enumerate(self.per_step_transformers_dicts):
            name = t.get("name")
            if name and name in per_wp_names:
                self.per_step_transformers_dicts[i] = per_wp_names[name]

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
            "tab_power": self.tab_power,
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
        if self.visible != visible:
            self.visible = visible
            self.visibility_changed.send(self)
            self.updated.send(self)

    def set_power(self, power: float):
        if not (0.0 <= power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
        if self.power != power:
            self.power = power
            self.updated.send(self)

    def set_cut_speed(self, speed: int):
        if self.cut_speed != speed:
            self.cut_speed = int(speed)
            self.updated.send(self)

    def set_travel_speed(self, speed: int):
        if self.travel_speed != speed:
            self.travel_speed = int(speed)
            self.updated.send(self)

    def set_air_assist(self, enabled: bool):
        if self.air_assist != enabled:
            self.air_assist = bool(enabled)
            self.updated.send(self)

    def set_kerf_mm(self, kerf: float):
        """Sets the kerf (beam width) in millimeters for this process."""
        if self.kerf_mm != kerf:
            self.kerf_mm = float(kerf)
            self.updated.send(self)

    def set_tab_power(self, power: float):
        if not (0.0 <= power <= 1.0):
            raise ValueError("Tab power must be between 0.0 and 1.0")
        if self.tab_power != power:
            self.tab_power = power
            self.updated.send(self)

    def get_operation_mode_short(self) -> Optional[str]:
        return None

    def get_summary(self) -> str:
        power_percent = round(self.power * 100)
        speed_str = format_value(self.cut_speed, "speed")
        return _("{power_percent}% power, {speed_str}").format(
            power_percent=power_percent, speed_str=speed_str
        )

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)

    def is_position_sensitive(self) -> bool:
        """
        Returns True if workpiece position changes may affect the output.

        This is true when per-workpiece transformers are configured that
        depend on the workpiece's world position (e.g., crop-to-stock).
        """
        for t_dict in self.per_workpiece_transformers_dicts:
            if not t_dict.get("enabled", True):
                continue
            name = t_dict.get("name")
            if not name or not isinstance(name, str):
                continue
            transformer_cls = transformer_registry.get(name)
            if transformer_cls is None:
                continue
            if transformer_cls.POSITION_SENSITIVE:
                return True
        return False
