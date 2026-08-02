from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.geo import Matrix
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.wavefront import AdaptiveWavefrontSpec
from raygeo.ops.part import Part

from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.varset import FloatVar, VarSet
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.transformer.registry import transformer_registry

from ..capabilities import CUT
from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine


class WavefrontStep(LaserStep):
    TYPELABEL = _("Wavefront")
    ICON = "step-wavefront-symbolic"
    CAPABILITIES: Tuple[StepCapability, ...] = (CUT,)
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "wavefront"

    RECIPE_KEYS: Tuple[str, ...] = LaserStep.RECIPE_KEYS + (
        "step_over_mm",
        "offset_mm",
    )

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *LaserStep.recipe_varset().vars,
                FloatVar(
                    key="step_over_mm",
                    label=_("Step Over"),
                    description=_(
                        "Distance between wavefront passes; defaults to "
                        "the laser spot width when unset"
                    ),
                    default=None,
                    min_val=0.0,
                ),
                FloatVar(
                    key="offset_mm",
                    label=_("Offset"),
                    default=0.0,
                ),
            ]
        )

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.8
        self.step_over_mm: Optional[float] = None
        self.offset_mm = 0.0
        self.area_tolerance = 0.01

    def get_assembler_kwargs(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> dict:
        spot_x, _spot_y = LaserHead.get_spot_size(
            self.get_selected_laser(machine)
        )
        kwargs: dict = {}
        kwargs["offset_mm"] = self.offset_mm
        kwargs["area_tolerance"] = self.area_tolerance
        kwargs["step_over"] = (
            self.step_over_mm if self.step_over_mm is not None else spot_x
        )
        kwargs["precision"] = machine.arc_tolerance
        kwargs["cut_feed_rate"] = self.cut_speed
        kwargs["cut_power"] = self.power
        return kwargs

    def build_compute_payload(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> "Tuple[Part, ComputePayload]":
        """Build a :class:`Part` with normalised-winding vector
        geometry and a :class:`ComputePayload` carrying an
        :class:`AdaptiveWavefrontSpec`."""
        part = _build_wavefront_part(workpiece)
        kwargs = self.get_assembler_kwargs(machine, workpiece)
        spec = AdaptiveWavefrontSpec(
            kwargs["step_over"],
            0.0,
            kwargs["area_tolerance"],
            kwargs["precision"],
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: "Machine",
        workpiece: "WorkPiece",
    ) -> Optional[dict]:
        return self.get_assembler_kwargs(machine, workpiece)

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["step_over_mm"] = self.step_over_mm
        result["offset_mm"] = self.offset_mm
        result["area_tolerance"] = self.area_tolerance
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "WavefrontStep":
        step = cast("WavefrontStep", super().from_dict(data))
        step.step_over_mm = data.get("step_over_mm", None)
        step.offset_mm = data.get("offset_mm", 0.0)
        step.area_tolerance = data.get("area_tolerance", 0.01)
        return step

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
        CropTransformer = transformer_registry.get("CropTransformer")
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        assert CropTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            CropTransformer(enabled=False).to_dict(),
            optimize_dict,
        ], [
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "WavefrontStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_laser_head()
        if default_head is None:
            raise ValueError("Machine has no laser heads configured.")

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()
        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_head_uid = default_head.uid
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
        return step


def _build_wavefront_part(workpiece: "WorkPiece") -> Part:
    """Build a :class:`Part` with normalised-winding vector geometry
    for the wavefront assembler.

    Disjoint pockets are exposed as separate faces via
    ``Part.from_geometry_multi_face`` so the per-face compute path
    clears each pocket independently (the wavefront assembler is a
    single-face operation since raygeo 1.28).

    Mirrors :func:`build_part_vector` with ``normalize_windings=True``
    but lives here so the compute-payload construction is
    self-contained for the raygeo intent pipeline.
    """
    boundaries = workpiece.boundaries
    if boundaries is None or boundaries.is_empty():
        return Part(size_mm=workpiece.size)
    scaled = boundaries.copy()
    w, h = workpiece.size
    if w > 0 and h > 0:
        scaled.transform(Matrix.scale(w, h))
    scaled.normalize_winding_orders()
    return Part.from_geometry_multi_face(geometry=scaled, size_mm=(w, h))
