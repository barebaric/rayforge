from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.contour import ContourSpec
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.cut_side import CutOrder, CutSide
from rayforge.core.varset import (
    BoolVar,
    LabeledChoiceVar,
    LengthVar,
    VarSet,
)
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector_with_raster_fallback,
)
from rayforge.pipeline.transformer.registry import transformer_registry

from ..capabilities import CUT, SCORE
from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class ContourStep(LaserStep):
    TYPELABEL = _("Contour")
    ICON = "step-contour-symbolic"
    CAPABILITIES: Tuple[StepCapability, ...] = (CUT, SCORE)
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "contour"

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *LaserStep.recipe_varset().vars,
                LabeledChoiceVar(
                    key="cut_side",
                    label=_("Cut Side"),
                    choices=[(cs.label(), cs.name) for cs in CutSide],
                    default="CENTERLINE",
                ),
                LabeledChoiceVar(
                    key="cut_order",
                    label=_("Cut Order"),
                    choices=[(co.label(), co.name) for co in CutOrder],
                    default="INSIDE_OUTSIDE",
                ),
                BoolVar(
                    key="remove_inner_paths",
                    label=_("Remove Inner Paths"),
                    default=False,
                ),
                LengthVar(
                    key="offset_mm",
                    label=_("Offset"),
                    description=_(
                        "Shifts the cut path inward/outward per Cut "
                        "Side (none on Centerline). Defaults to kerf "
                        "compensation for the head"
                    ),
                    default=0.0,
                ),
                LengthVar(
                    key="overcut",
                    label=_("Overcut"),
                    default=0.0,
                    min_val=0.0,
                ),
            ]
        )

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.8
        self.cut_side = "CENTERLINE"
        self.cut_order = "INSIDE_OUTSIDE"
        self.remove_inner_paths = False
        self.offset_mm = 0.0
        self.overcut = 0.0
        self.override_threshold = False
        self.threshold = 0.5

    def get_operation_mode_short(self):
        try:
            return CutSide[self.cut_side].label()
        except (KeyError, TypeError):
            return None

    def get_assembler_kwargs(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = str(self.cut_side).lower()
        kwargs["cut_order"] = str(self.cut_order).lower()
        kwargs["remove_inner"] = self.remove_inner_paths
        kwargs["offset_mm"] = self.offset_mm
        kwargs["overcut"] = self.overcut
        kwargs["arc_tolerance"] = machine.arc_tolerance
        kwargs["allow_arcs"] = machine.supports_arcs
        kwargs["supports_curves"] = machine.supports_curves
        return kwargs

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> "Tuple[Part, ComputePayload]":
        """Build a :class:`Part` (from the workpiece's vector
        geometry) and a :class:`ComputePayload` carrying a
        :class:`ContourSpec` populated from this step's resolved
        assembler kwargs.

        When the workpiece has no vector boundaries (e.g. an SVG
        with empty ``pristine_geometry``), the source is rendered
        to pixels and traced into geometry before assembling.
        """
        part = build_part_vector_with_raster_fallback(
            workpiece,
            self.pixels_per_mm,
            override_threshold=self.override_threshold,
            threshold=self.threshold,
        )
        kwargs = self.get_assembler_kwargs(machine, workpiece)
        spec = ContourSpec(
            offset_mm=kwargs["offset_mm"],
            cut_side=kwargs["cut_side"],
            overcut=kwargs["overcut"],
            cut_order=kwargs["cut_order"],
            remove_inner=kwargs["remove_inner"],
            arc_tolerance=kwargs["arc_tolerance"],
            allow_arcs=kwargs["allow_arcs"],
            supports_curves=kwargs["supports_curves"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> Optional[dict]:
        """Expose the resolved assembler kwargs for the compute token."""
        return self.get_assembler_kwargs(machine, workpiece)

    def apply_import_settings(self, settings: dict) -> None:
        """Apply importer-provided settings this step owns."""
        super().apply_import_settings(settings)
        offset_mm = settings.get("offset_mm")
        if offset_mm is not None:
            self.offset_mm = offset_mm

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["cut_side"] = self.cut_side
        data["cut_order"] = self.cut_order
        data["remove_inner_paths"] = self.remove_inner_paths
        data["offset_mm"] = self.offset_mm
        data["overcut"] = self.overcut
        data["override_threshold"] = self.override_threshold
        data["threshold"] = self.threshold
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ContourStep":
        step = cast("ContourStep", super().from_dict(data))
        step.cut_side = data.get("cut_side", "CENTERLINE")
        step.cut_order = data.get("cut_order", "INSIDE_OUTSIDE")
        step.remove_inner_paths = data.get("remove_inner_paths", False)
        if "offset_mm" in data:
            step.offset_mm = data["offset_mm"]
        else:
            step.offset_mm = data.get("path_offset_mm", 0.0) + (
                data.get("kerf_mm", 0.0) / 2.0
            )
        step.overcut = data.get("overcut", 0.0)
        step.override_threshold = data.get("override_threshold", False)
        step.threshold = data.get("threshold", 0.5)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        Smooth = transformer_registry.get("Smooth")
        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        TabOpsTransformer = transformer_registry.get("TabOpsTransformer")
        CropTransformer = transformer_registry.get("CropTransformer")
        MergeLinesTransformer = transformer_registry.get(
            "MergeLinesTransformer"
        )
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert Smooth is not None
        assert LeadInOutTransformer is not None
        assert TabOpsTransformer is not None
        assert CropTransformer is not None
        assert MergeLinesTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            Smooth(enabled=False, amount=20).to_dict(),
            LeadInOutTransformer(
                enabled=False, lead_in_mm=0, lead_out_mm=0, auto=True
            ).to_dict(),
            TabOpsTransformer().to_dict(),
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            MergeLinesTransformer().to_dict(),
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        optimize: bool = True,
        **kwargs,
    ) -> "ContourStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_laser_head()
        if default_head is None:
            raise ValueError("Machine has no laser heads configured.")

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()
        if not optimize:
            per_wp = [t for t in per_wp if t.get("name") != "Optimize"]

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_head_uid = default_head.uid
        step.offset_mm = default_head.kerf_mm
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        # Operating feed defaults are machine-derived: the machine only
        # exposes its ceiling, so the default is that ceiling, bounded by
        # the operation's typical feed rate.
        step.cut_speed = min(machine.max_cut_speed, 500)
        params = machine.get_pwm_params(default_head)
        if params is not None:
            step.frequency = params.frequency
            step.pulse_width = params.pulse_width

        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(step.cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        return step
