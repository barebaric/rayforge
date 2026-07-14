from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, cast

from rayforge.core.capability import MATERIAL_TEST, Capability
from rayforge.core.step import Step
from rayforge.pipeline.stage.assembler_helpers import MachineDefaults
from rayforge.pipeline.transformer.registry import transformer_registry

from ..producers import MaterialTestGridProducer

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class MaterialTestStep(Step):
    TYPELABEL = _("Material Test Grid")
    ICON = "test-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (MATERIAL_TEST,)
    PRODUCER_CLASS = MaterialTestGridProducer
    ASSEMBLER_NAME = "material_test_grid"
    HIDDEN = True

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.test_type = "Cut"
        self.grid_mode = "Power vs Speed"
        self.speed_range = (100.0, 500.0)
        self.power_range = (10.0, 100.0)
        self.passes_range = (1, 5)
        self.fixed_speed = 1000.0
        self.fixed_power = 50.0
        self.grid_dimensions = (5, 5)
        self.shape_size = 10.0
        self.spacing = 2.0
        self.include_labels = True
        self.label_power_percent = 10.0
        self.label_speed = 1000.0
        self.line_interval_mm = None

    def get_assembler_kwargs(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        kwargs: dict = {}
        kwargs["size_mm"] = workpiece.size if workpiece else (0, 0)
        kwargs["cols"] = self.grid_dimensions[0]
        kwargs["rows"] = self.grid_dimensions[1]
        kwargs["min_speed"] = self.speed_range[0]
        kwargs["max_speed"] = self.speed_range[1]
        kwargs["min_power"] = self.power_range[0]
        kwargs["max_power"] = self.power_range[1]
        kwargs["min_passes"] = self.passes_range[0]
        kwargs["max_passes"] = self.passes_range[1]
        kwargs["mode"] = "cut" if self.test_type == "Cut" else "engrave"
        kwargs["grid_mode"] = self.grid_mode
        kwargs["fixed_speed"] = self.fixed_speed
        kwargs["fixed_power"] = self.fixed_power
        kwargs["shape_size"] = self.shape_size
        kwargs["spacing"] = self.spacing
        kwargs["include_labels"] = self.include_labels
        kwargs["line_interval_mm"] = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else machine_defaults.line_interval_mm
        )
        return kwargs

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["test_type"] = self.test_type
        result["grid_mode"] = self.grid_mode
        result["speed_range"] = list(self.speed_range)
        result["power_range"] = list(self.power_range)
        result["passes_range"] = list(self.passes_range)
        result["fixed_speed"] = self.fixed_speed
        result["fixed_power"] = self.fixed_power
        result["grid_dimensions"] = list(self.grid_dimensions)
        result["shape_size"] = self.shape_size
        result["spacing"] = self.spacing
        result["include_labels"] = self.include_labels
        result["label_power_percent"] = self.label_power_percent
        result["label_speed"] = self.label_speed
        result["line_interval_mm"] = self.line_interval_mm
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialTestStep":
        step = cast("MaterialTestStep", super().from_dict(data))
        step.test_type = data.get("test_type", "Cut")
        step.grid_mode = data.get("grid_mode", "Power vs Speed")
        step.speed_range = tuple(data.get("speed_range", (100.0, 500.0)))
        step.power_range = tuple(data.get("power_range", (10.0, 100.0)))
        step.passes_range = tuple(data.get("passes_range", (1, 5)))
        step.fixed_speed = data.get("fixed_speed", 1000.0)
        step.fixed_power = data.get("fixed_power", 50.0)
        step.grid_dimensions = tuple(data.get("grid_dimensions", (5, 5)))
        step.shape_size = data.get("shape_size", 10.0)
        step.spacing = data.get("spacing", 2.0)
        step.include_labels = data.get("include_labels", True)
        step.label_power_percent = data.get("label_power_percent", 10.0)
        step.label_speed = data.get("label_speed", 1000.0)
        step.line_interval_mm = data.get("line_interval_mm", None)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        OverscanTransformer = transformer_registry.get("OverscanTransformer")
        Optimize = transformer_registry.get("Optimize")
        assert OverscanTransformer is not None
        assert Optimize is not None
        optimize_dict = Optimize().to_dict()
        return [
            OverscanTransformer(
                enabled=True, distance_mm=0, auto=True
            ).to_dict(),
            optimize_dict,
        ], [
            optimize_dict,
        ]

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
        default_dict = cls.PRODUCER_CLASS().to_dict()
        step.opsproducer_dict = default_dict
        per_wp, per_step = cls.get_default_transformers_dicts()

        OverscanTransformer = cast(
            "OverscanTransformerType",
            transformer_registry.get("OverscanTransformer"),
        )
        assert OverscanTransformer is not None
        auto_distance = OverscanTransformer.calculate_auto_distance(
            step.cut_speed, machine.acceleration
        )
        for t in per_wp:
            if t.get("name") == "OverscanTransformer":
                t["distance_mm"] = auto_distance

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        default_head = machine.get_default_head()
        step.selected_laser_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)
        return step
