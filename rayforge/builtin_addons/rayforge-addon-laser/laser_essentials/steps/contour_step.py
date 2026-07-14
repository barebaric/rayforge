from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

from rayforge.core.capability import CUT, SCORE, WITH_KERF, Capability
from rayforge.core.step import Step
from rayforge.pipeline.producer.base import CutSide
from rayforge.pipeline.stage.assembler_helpers import MachineDefaults
from rayforge.pipeline.transformer.registry import transformer_registry

from ..producers import ContourProducer

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece


class ContourStep(Step):
    TYPELABEL = _("Contour")
    ICON = "step-contour-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (CUT, SCORE, WITH_KERF)
    PRODUCER_CLASS = ContourProducer
    ASSEMBLER_NAME = "contour"

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.cut_side = "CENTERLINE"
        self.cut_order = "INSIDE_OUTSIDE"
        self.remove_inner_paths = False
        self.path_offset_mm = 0.0
        self.overcut = 0.0

    def get_operation_mode_short(self):
        try:
            return CutSide[self.cut_side].label()
        except (KeyError, TypeError):
            return None

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["cut_side"] = str(self.cut_side).lower()
        kwargs["cut_order"] = str(self.cut_order).lower()
        kwargs["remove_inner"] = self.remove_inner_paths
        kwargs["path_offset_mm"] = self.path_offset_mm
        kwargs["overcut"] = self.overcut
        kwargs["kerf_mm"] = machine_defaults.kerf_mm
        kwargs["arc_tolerance"] = machine_defaults.arc_tolerance
        kwargs["allow_arcs"] = machine_defaults.allow_arcs
        kwargs["supports_curves"] = machine_defaults.supports_curves
        return kwargs

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["cut_side"] = self.cut_side
        data["cut_order"] = self.cut_order
        data["remove_inner_paths"] = self.remove_inner_paths
        data["path_offset_mm"] = self.path_offset_mm
        data["overcut"] = self.overcut
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ContourStep":
        step = cast("ContourStep", super().from_dict(data))
        step.cut_side = data.get("cut_side", "CENTERLINE")
        step.cut_order = data.get("cut_order", "INSIDE_OUTSIDE")
        step.remove_inner_paths = data.get("remove_inner_paths", False)
        step.path_offset_mm = data.get("path_offset_mm", 0.0)
        step.overcut = data.get("overcut", 0.0)
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
        default_head = machine.get_default_head()

        step = cls(name=name)
        default_dict = cls.PRODUCER_CLASS().to_dict()
        step.opsproducer_dict = default_dict
        per_wp, per_step = cls.get_default_transformers_dicts()
        if not optimize:
            per_wp = [t for t in per_wp if t.get("name") != "Optimize"]

        LeadInOutTransformer = transformer_registry.get("LeadInOutTransformer")
        if LeadInOutTransformer:
            calc = getattr(LeadInOutTransformer, "calculate_auto_distance")
            auto_distance = calc(machine.max_cut_speed, machine.acceleration)
            for t in per_wp:
                if t.get("name") == "LeadInOutTransformer":
                    t["lead_in_mm"] = auto_distance
                    t["lead_out_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.kerf_mm = default_head.spot_size_mm[0]
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
