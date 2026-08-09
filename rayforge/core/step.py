import logging
from abc import ABC
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    cast,
)

from blinker import Signal
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Matrix
from raygeo.ops import Ops
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part
from raygeo.ops.state import CoolantMode

from ..machine.models.head import Head
from ..machine.models.spindle import SpindleHead
from ..pipeline.transformer.registry import transformer_registry
from .capability import MachineCapability, StepCapability
from .item import DocItem
from .step_registry import step_registry
from .varset import SpeedVar, VarSet

if TYPE_CHECKING:
    from ..context import RayforgeContext
    from ..machine.models.machine import Machine
    from .layer import Layer
    from .workflow import Workflow
    from .workpiece import WorkPiece


logger = logging.getLogger(__name__)

_COOLANT_MODE_BY_NAME = {
    mode.name: mode
    for mode in (CoolantMode.OFF, CoolantMode.FLOOD, CoolantMode.MIST)
}


def legacy_producer_params(data: dict[str, Any]) -> dict[str, Any]:
    """Return the legacy ``opsproducer_dict.params`` payload, if any.

    Projects saved before the raygeo-pipeline refactor stored each
    step's producer configuration under ``opsproducer_dict``.  Step
    ``from_dict`` implementations consult this payload so the saved
    parameters survive loading; current-format top-level keys always
    take precedence.
    """
    opsproducer = data.get("opsproducer_dict")
    if isinstance(opsproducer, dict):
        params = opsproducer.get("params")
        if isinstance(params, dict):
            return params
    return {}


class Step(DocItem, ABC):
    """
    An OpsProducer configuration that operates on WorkPieces.

    A Step is a stateless configuration object that defines a single
    operation (e.g., outline, engrave) to be performed. It holds its
    configuration as serializable dictionaries.
    """

    HIDDEN: bool = False
    ICON: str = ""
    CAPABILITIES: tuple[StepCapability, ...] = ()
    REQUIRED_MACHINE_CAPS: ClassVar[frozenset[MachineCapability]] = frozenset()
    ASSEMBLER_NAME: ClassVar[str] = ""
    uses_global_state: ClassVar[bool] = False

    def __init__(
        self,
        typelabel: str,
        name: str | None = None,
    ):
        super().__init__(name=name or typelabel)
        self.typelabel = typelabel
        self.visible = True
        self.selected_head_uid: str | None = None
        self.generated_workpiece_uid: str | None = None
        self.applied_recipe_uid: str | None = None

        per_wp_defaults, per_sp_defaults = (
            self.get_default_transformers_dicts()
        )
        self.per_workpiece_transformers_dicts: list[dict[str, Any]] = list(
            per_wp_defaults
        )
        self.per_step_transformers_dicts: list[dict[str, Any]] = list(
            per_sp_defaults
        )

        self.pixels_per_mm = 50, 50

        # Signals for notifying of model changes
        self.per_step_transformer_changed = Signal()
        self.visibility_changed = Signal()

        # Default machine-dependent values.
        self.cut_speed: int = 500
        self.max_cut_speed = 10000
        self.travel_speed: int = 5000
        self.max_travel_speed = 10000

        # Coolant method used while this step runs.
        self.coolant_method: CoolantMode = CoolantMode.OFF

        # Forward compatibility: store unknown attributes
        self.extra: dict[str, Any] = {}

        # Set when a step of an unknown type is deserialized, so the
        # original type name can be reported and round-tripped.
        self._original_step_type: str | None = None

    @property
    def capabilities(self) -> tuple[StepCapability, ...]:
        return type(self).CAPABILITIES

    @classmethod
    def recipe_varset(cls) -> VarSet:
        """The VarSet used to render this step type's recipe editor.

        The base returns the shared motion vars. Domain bases and
        concrete steps extend this (via ``super()`` composition) with
        their own attributes. Recipe extraction keys are derived from
        this varset via :meth:`recipe_keys`, so the editor and the
        extractor agree.
        """
        return VarSet(
            vars=[
                SpeedVar(
                    key="cut_speed",
                    label=_("Cut Speed"),
                    default=500,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                    role="travel",
                ),
            ]
        )

    @classmethod
    def recipe_keys(cls) -> tuple[str, ...]:
        """The step attribute keys eligible for recipe extraction.

        Derived from :meth:`recipe_varset` so the editor and the
        extractor always agree on which attributes a step's recipe
        carries. Domain bases and concrete steps inherit this through
        the same ``super()`` composition as :meth:`recipe_varset`.
        """
        return tuple(var.key for var in cls.recipe_varset())

    @classmethod
    def recipe_varset_groups(cls) -> list[tuple[str, VarSet]]:
        """Split :meth:`recipe_varset` into named groups for the editor.

        Returns a list of ``(title, varset)`` pairs. The base returns a
        single group. Domain bases override this to separate inherited
        process settings from step-specific settings (e.g. "Laser" vs
        "Step Settings").
        """
        return [(_("Settings"), cls.recipe_varset())]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: str | None = None,
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

    def get_assembler_kwargs(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> dict[str, Any]:
        """Build the kwargs dict for :meth:`~.AssemblerRegistry.assemble`."""
        return {}

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> "tuple[Part, ComputePayload]":
        """
        Build the raygeo :class:`Part` and :class:`ComputePayload` for
        a workpiece compute node of the new intent pipeline.

        The base implementation returns a default payload wrapping a
        bare :class:`ContourSpec` assembler and a :class:`Part`
        built from the workpiece's vector geometry (or an empty
        :class:`Part` when the workpiece has no boundaries).  Step
        kinds with a real raygeo assembler override this to populate
        the assembler spec from their own machine resolution (see
        :class:`ContourStep`, :class:`EngraveStep`).

        :param machine: The machine context the step resolves its
            process defaults from.
        :param workpiece: The workpiece this compute node runs against.
        :returns: ``(part, payload)`` for ``StageSpec.Compute``.
        """
        part = workpiece.to_part()
        if part is None:
            part = Part(size_mm=workpiece.size)
        return part, ComputePayload(assembler=Assembler(ContourSpec()))

    def assembler_token_params(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> dict[str, Any] | None:
        """
        Return a JSON-serialisable dict of the assembler spec
        parameters that this step resolves for *machine*.

        The value is folded into the workpiece compute token so that
        changes to step-specific assembler inputs (e.g. ``cut_side``
        for ContourStep) invalidate the cache even when the generic
        step parameters are unchanged.

        The base implementation returns :data:`None`, leaving the
        compute token unaffected.  Step kinds that wire a real
        assembler spec override this (see :class:`ContourStep`).
        """
        return None

    def populate_payload(self, payload, machine: "Machine"):
        """Set domain-specific fields on the ComputePayload.

        The base stamps the shared motion fields and the resolved head
        uid, leaving the process power at its neutral default. Domain
        bases override this to add their own process fields (e.g. laser
        power) and never read attributes they do not own.
        """
        payload.cut_speed = self.cut_speed
        head = self.get_selected_head(machine)
        payload.head_uid = head.uid if head else None
        payload.power = 0.0

    def get_cache_params(self) -> dict[str, Any]:
        """JSON-serialisable step attributes that influence compute output.

        UIDs and cosmetic fields are intentionally omitted so the token
        only changes when the actual compute inputs change. Domain bases
        extend this with their own process attributes.
        """
        return {
            "type": type(self).__name__,
            "visible": self.visible,
            "cut_speed": self.cut_speed,
            "max_cut_speed": self.max_cut_speed,
            "travel_speed": self.travel_speed,
            "max_travel_speed": self.max_travel_speed,
            "coolant_method": self.coolant_method.name,
            "pixels_per_mm": list(self.pixels_per_mm),
        }

    def create_initial_ops(self) -> "Ops":
        """Build the initial Ops object with step-wide machine settings.

        The generic step has no process parameters of its own; domain
        bases (e.g. :class:`LaserStep`) override this to stamp their
        machine settings.
        """
        ops = Ops()
        if self.coolant_method is not CoolantMode.OFF:
            ops.set_coolant(self.coolant_method)
        return ops

    def apply_import_settings(self, settings: dict[str, Any]) -> None:
        """Apply importer-provided settings that this step owns.

        The settings dict uses the step's own attribute names
        (canonicalised by the importer). The base handles the shared
        motion settings; domain bases override this to apply their own
        process attributes and call ``super()``.
        """
        cut_speed = settings.get("cut_speed")
        if cut_speed is not None:
            self.set_cut_speed(cut_speed)

    def to_dict(self) -> dict:
        """Serializes the step and its configuration to a dictionary."""
        step_type = (
            self._original_step_type
            if self._original_step_type is not None
            else self.__class__.__name__
        )
        result = {
            "uid": self.uid,
            "type": "step",
            "step_type": step_type,
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "typelabel": self.typelabel,
            "visible": self.visible,
            "selected_head_uid": self.selected_head_uid,
            "generated_workpiece_uid": self.generated_workpiece_uid,
            "applied_recipe_uid": self.applied_recipe_uid,
            "per_workpiece_transformers_dicts": (
                self.per_workpiece_transformers_dicts
            ),
            "per_step_transformers_dicts": self.per_step_transformers_dicts,
            "pixels_per_mm": self.pixels_per_mm,
            "cut_speed": self.cut_speed,
            "max_cut_speed": self.max_cut_speed,
            "travel_speed": self.travel_speed,
            "max_travel_speed": self.max_travel_speed,
            "coolant_method": self.coolant_method.name,
            "children": [child.to_dict() for child in self.children],
        }
        result.update(self.extra)
        return result

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        """Keys this class handles in ``to_dict``/``from_dict``.

        Used solely for ``extra``-dict filtering: unknown keys from
        newer file versions are preserved in ``extra`` rather than
        silently dropped. Subclasses that serialize additional keys
        extend this via ``super()`` composition in the MRO.
        """
        return frozenset(
            {
                "uid",
                "type",
                "step_type",
                "name",
                "matrix",
                "typelabel",
                "visible",
                "selected_laser_uid",
                "selected_head_uid",
                "generated_workpiece_uid",
                "applied_recipe_uid",
                "modifiers_dicts",
                "per_workpiece_transformers_dicts",
                "per_step_transformers_dicts",
                "pixels_per_mm",
                "cut_speed",
                "max_cut_speed",
                "travel_speed",
                "max_travel_speed",
                "coolant_method",
                "children",
            }
        )

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        """
        Returns default transformer configurations for this step type.

        Returns:
            A tuple of (per_workpiece_transformers_dicts,
            per_step_transformers_dicts) for new steps of this type.
            Subclasses should override this to provide their defaults.
        """
        return [], []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        """Deserializes a Step instance from a dictionary."""
        extra = {
            k: v for k, v in data.items() if k not in cls._serialized_keys()
        }

        step_type_name = data.get("step_type")
        if step_type_name:
            step_class = step_registry.get(step_type_name)
        else:
            step_class = None

        if step_class is None:
            typelabel = data.get("typelabel")
            if typelabel:
                step_class = step_registry.get_by_typelabel(typelabel)

        if step_class is not None and step_class is not cls:
            return step_class.from_dict(data)

        if step_class is None:
            step_class = cls
            # Preserve the original step type name so a missing step
            # can be reported and round-tripped when the addon
            # providing it is not installed.
            original_step_type = step_type_name
        else:
            original_step_type = None

        step = step_class(typelabel=data["typelabel"], name=data.get("name"))
        if original_step_type:
            step._original_step_type = original_step_type
        step.uid = data["uid"]
        step.matrix = Matrix.from_list(data["matrix"])
        step.visible = data["visible"]
        step.selected_head_uid = data.get(
            "selected_head_uid", data.get("selected_laser_uid")
        )
        step.generated_workpiece_uid = data.get("generated_workpiece_uid")
        step.applied_recipe_uid = data.get("applied_recipe_uid")

        default_per_wp, default_per_step = (
            step_class.get_default_transformers_dicts()
        )
        step.per_workpiece_transformers_dicts = Step._merge_transformer_dicts(
            data.get("per_workpiece_transformers_dicts", []),
            default_per_wp,
        )
        step.per_step_transformers_dicts = Step._merge_transformer_dicts(
            data.get("per_step_transformers_dicts", []),
            default_per_step,
        )

        # Share dict references for transformers that appear in both lists
        step._unify_shared_transformers()

        step.pixels_per_mm = data.get("pixels_per_mm", (100, 100))
        step.max_cut_speed = data.get("max_cut_speed", step.max_cut_speed)
        step.max_travel_speed = data.get(
            "max_travel_speed", step.max_travel_speed
        )
        step.cut_speed = data.get("cut_speed", step.cut_speed)
        step.travel_speed = data.get("travel_speed", step.travel_speed)
        raw_coolant = data.get("coolant_method", CoolantMode.OFF.name)
        step.coolant_method = _COOLANT_MODE_BY_NAME.get(
            raw_coolant, CoolantMode.OFF
        )
        step.extra = extra
        return step

    @staticmethod
    def _merge_transformer_dicts(
        loaded: list[dict[str, Any]], defaults: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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

    @property
    def original_step_type(self) -> str | None:
        """
        The step type name stored in the source document.

        When a step's class is not registered (e.g. because the addon
        providing it is not installed), ``from_dict`` preserves the
        original ``step_type`` so the missing feature can be reported
        and round-tripped. Registered steps return ``None``.
        """
        return self._original_step_type

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

    def get_selected_head(self, machine: "Machine") -> Head | None:
        """
        Resolves and returns the selected head for this step, or None
        if the machine has no heads. Falls back to the first head on
        the machine if the selection is invalid or not set.
        """
        if self.selected_head_uid:
            for head in machine.heads:
                if head.uid == self.selected_head_uid:
                    return head
        # Fallback
        if machine.heads:
            return machine.heads[0]
        return None

    def set_selected_head_uid(self, uid: str | None):
        """
        Sets the UID of the head to be used by this step.
        """
        if self.selected_head_uid != uid:
            self.selected_head_uid = uid
            self.updated.send(self)

    def set_name(self, name: str):
        if self.name != name:
            self.name = name

    def set_visible(self, visible: bool):
        if self.visible != visible:
            self.visible = visible
            self.visibility_changed.send(self)
            self.updated.send(self)

    def set_cut_speed(self, speed: int):
        if self.cut_speed != speed:
            self.cut_speed = int(speed)
            self.updated.send(self)

    def set_travel_speed(self, speed: int):
        if self.travel_speed != speed:
            self.travel_speed = int(speed)
            self.updated.send(self)

    def set_coolant_method(self, mode: CoolantMode):
        """Sets the coolant method used while this step runs."""
        if self.coolant_method is not mode:
            self.coolant_method = mode
            self.updated.send(self)

    def get_unsupported_coolant_methods(
        self, machine: "Machine"
    ) -> tuple[CoolantMode, ...]:
        """Coolant methods this step uses that the machine's selected
        head does not support.

        ``CoolantMode.OFF`` is always supported, so it is never
        reported. Non-spindle heads (e.g. laser heads) have no coolant
        methods, so nothing is reported for them either.
        """
        if self.coolant_method is CoolantMode.OFF:
            return ()
        head = self.get_selected_head(machine)
        if not isinstance(head, SpindleHead):
            return ()
        if self.coolant_method in head.cooling_methods:
            return ()
        return (self.coolant_method,)

    def get_operation_mode_short(self) -> str | None:
        return None

    def get_operation_color(self, head) -> str | None:
        """Return the color used to represent this step's operation for
        the given head, or None when the step has no color.

        Domain bases override this (e.g. laser steps return the head's
        raster or cut color).
        """
        return None

    def get_summary(self) -> str:
        """Return a short human-readable summary for the UI.

        The generic step has no process parameters of its own, so it
        falls back to the type label. Domain bases (e.g.
        :class:`LaserStep`) override this to describe their process.
        """
        return self.typelabel

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
