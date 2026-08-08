from __future__ import annotations

from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
)

import numpy as np
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.raster import RasterSpec
from raygeo.ops.part import Part
from raygeo.ops.part.image_source import WholeImageSource

from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.step import legacy_producer_params
from rayforge.core.varset import (
    BoolVar,
    FloatVar,
    LabeledChoiceVar,
    SliderFloatVar,
    VarSet,
)
from rayforge.image.dither import DitherAlgorithm
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    compute_raster_auto_levels,
    preprocess_raster_image,
)
from rayforge.pipeline.transformer.registry import transformer_registry

from ..capabilities import ENGRAVE
from .laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class EngraveStep(LaserStep):
    TYPELABEL = _("Engrave")
    ICON = "step-raster-symbolic"
    CAPABILITIES: tuple[StepCapability, ...] = (ENGRAVE,)
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})
    ASSEMBLER_NAME = "raster"

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                *LaserStep.recipe_varset().vars,
                FloatVar(
                    key="scan_angle",
                    label=_("Scan Angle"),
                    default=0.0,
                    min_val=0.0,
                    max_val=360.0,
                ),
                LabeledChoiceVar(
                    key="depth_mode",
                    label=_("Depth Mode"),
                    choices=[(m.display_name, m.name) for m in DepthMode],
                    default="POWER_MODULATION",
                ),
                BoolVar(
                    key="invert",
                    label=_("Invert"),
                    default=False,
                ),
                SliderFloatVar(
                    key="min_power_level",
                    label=_("Min Power Level"),
                    default=0.0,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
                SliderFloatVar(
                    key="max_power_level",
                    label=_("Max Power Level"),
                    default=1.0,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
            ]
        )

    def __init__(self, name: str | None = None, typelabel: str | None = None):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.power = 0.2
        self.scan_angle = 0.0
        self.depth_mode = "POWER_MODULATION"
        self.invert = False
        self.auto_levels = True
        self.black_point = 0
        self.white_point = 255
        self.threshold = 128
        self.line_interval_mm = None
        self.sample_interval_mm = None
        self.dot_width_correction_mm = None
        self.min_power_level = 0.0
        self.max_power_level = 1.0
        self.num_power_levels = 25
        self.offset_x_mm = 0.0
        self.offset_y_mm = 0.0
        self.scan_mode = "SEGMENTED"
        self.cross_hatch = False
        self.num_depth_levels = 5
        self.z_step_down = 0.0
        self.angle_increment = 0.0
        self.dither_algorithm = None
        self.bidir_x_offset_mm = 0.0

    def get_operation_mode_short(self):
        if not self.depth_mode:
            return None
        try:
            return DepthMode[self.depth_mode].short_name
        except KeyError:
            return None

    def get_operation_color(self, head) -> str | None:
        """The head's raster color, used to represent engraving."""
        if isinstance(head, LaserHead):
            return head.raster_color
        return None

    def is_position_sensitive(self) -> bool:
        """The raster assembler bakes ``workpiece.bbox`` into its
        output via ``offset_x_mm`` / ``offset_y_mm`` so the compute
        result depends on the workpiece's absolute world position
        (not just on per-workpiece transformers like CropTransformer).
        Returning True ensures the compute token folds in
        ``transform_revision`` so a pure move invalidates the
        workpiece compute cache rather than leaving stale,
        wrong-position ops to be re-displaced by the aggregate's new
        placement matrix."""
        return True

    def get_assembler_kwargs(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict:
        _spot_x, spot_y = LaserHead.get_spot_size(
            self.get_selected_laser(machine)
        )
        line_interval = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else spot_y
        )
        return {
            "mode": DepthMode[self.depth_mode].raygeo_name,
            "line_interval_mm": line_interval,
            "sample_interval_mm": self.sample_interval_mm,
            "dot_width_correction_mm": self.dot_width_correction_mm,
            "min_power": self.min_power_level,
            "max_power": self.max_power_level,
            "step_power": self.power,
            "num_power_levels": self.num_power_levels,
            "angle": self.scan_angle,
            "offset_x_mm": self.offset_x_mm,
            "offset_y_mm": self.offset_y_mm,
            "scan_mode": self.scan_mode.lower(),
            "cross_hatch": self.cross_hatch,
            "num_depth_levels": self.num_depth_levels,
            "z_step_down": self.z_step_down,
            "angle_increment": self.angle_increment,
        }

    def apply_import_settings(self, settings: dict[str, Any]) -> None:
        """Apply importer-provided raster settings this step owns."""
        super().apply_import_settings(settings)
        for key in (
            "min_power_level",
            "max_power_level",
            "dot_width_correction_mm",
            "line_interval_mm",
            "scan_angle",
        ):
            if key in settings:
                setattr(self, key, settings[key])

    def build_compute_payload(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> tuple[Part, ComputePayload]:
        """Build a :class:`Part` with the preprocessed raster image
        attached as a :class:`WholeImageSource`, and a
        :class:`ComputePayload` carrying a :class:`RasterSpec`.

        Rendering and preprocessing (dither / auto-levels / depth
        mode) happen here, on the calling thread, so the Rust
        assembler on the rayon worker only reads slabs from the
        attached image source.
        """
        spot_x, spot_y = LaserHead.get_spot_size(
            self.get_selected_laser(machine)
        )
        part, alpha = _build_raster_part(self, machine, workpiece)
        kwargs = self.get_assembler_kwargs(machine, workpiece)
        depth_mode = DepthMode[self.depth_mode]
        line_interval = kwargs["line_interval_mm"] or spot_y
        sample_interval = kwargs["sample_interval_mm"] or spot_x / 2.0
        dot_width = (
            kwargs["dot_width_correction_mm"]
            if kwargs["dot_width_correction_mm"] is not None
            else spot_x / 2.0
        )
        x_off, y_off, _w, _h = workpiece.bbox
        alpha_arr = (
            (alpha * 255).astype(np.uint8).tobytes()
            if alpha is not None
            else None
        )
        spec = RasterSpec(
            mode=depth_mode.raygeo_name,
            line_interval_mm=line_interval,
            sample_interval_mm=sample_interval,
            min_power=kwargs["min_power"],
            max_power=kwargs["max_power"],
            step_power=kwargs["step_power"],
            num_power_levels=kwargs["num_power_levels"],
            angle=kwargs["angle"],
            offset_x_mm=x_off,
            offset_y_mm=y_off,
            scan_mode=kwargs["scan_mode"],
            cross_hatch=kwargs["cross_hatch"],
            num_depth_levels=kwargs["num_depth_levels"],
            z_step_down=kwargs["z_step_down"],
            angle_increment=kwargs["angle_increment"],
            dot_width_correction_mm=dot_width,
            alpha=alpha_arr,
        )
        return part, ComputePayload(assembler=Assembler(spec))

    def assembler_token_params(
        self,
        machine: Machine,
        workpiece: WorkPiece,
    ) -> dict | None:
        return self.get_assembler_kwargs(machine, workpiece)

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["scan_angle"] = self.scan_angle
        result["depth_mode"] = self.depth_mode
        result["invert"] = self.invert
        result["auto_levels"] = self.auto_levels
        result["black_point"] = self.black_point
        result["white_point"] = self.white_point
        result["threshold"] = self.threshold
        result["line_interval_mm"] = self.line_interval_mm
        result["sample_interval_mm"] = self.sample_interval_mm
        result["dot_width_correction_mm"] = self.dot_width_correction_mm
        result["min_power_level"] = self.min_power_level
        result["max_power_level"] = self.max_power_level
        result["num_power_levels"] = self.num_power_levels
        result["offset_x_mm"] = self.offset_x_mm
        result["offset_y_mm"] = self.offset_y_mm
        result["scan_mode"] = self.scan_mode
        result["cross_hatch"] = self.cross_hatch
        result["num_depth_levels"] = self.num_depth_levels
        result["z_step_down"] = self.z_step_down
        result["angle_increment"] = self.angle_increment
        result["dither_algorithm"] = (
            self.dither_algorithm.value if self.dither_algorithm else None
        )
        result["bidir_x_offset_mm"] = self.bidir_x_offset_mm
        return result

    @classmethod
    def from_dict(cls, data: dict) -> EngraveStep:
        step = cast("EngraveStep", super().from_dict(data))
        legacy = legacy_producer_params(data)
        # Legacy type names implied a depth mode when none was saved.
        old_type = data.get("opsproducer_dict", {}).get("type")
        if old_type == "Rasterizer" and "depth_mode" not in legacy:
            legacy["depth_mode"] = "CONSTANT_POWER"
            if "direction_degrees" in legacy:
                legacy["scan_angle"] = legacy.pop("direction_degrees")
        elif old_type == "DitherRasterizer":
            legacy["depth_mode"] = "DITHER"
        step.scan_angle = data.get("scan_angle", legacy.get("scan_angle", 0.0))
        step.depth_mode = data.get(
            "depth_mode", legacy.get("depth_mode", "POWER_MODULATION")
        )
        step.invert = data.get("invert", legacy.get("invert", False))
        step.auto_levels = data.get(
            "auto_levels", legacy.get("auto_levels", True)
        )
        step.black_point = data.get(
            "black_point", legacy.get("black_point", 0)
        )
        step.white_point = data.get(
            "white_point", legacy.get("white_point", 255)
        )
        step.threshold = data.get("threshold", legacy.get("threshold", 128))
        step.line_interval_mm = data.get(
            "line_interval_mm", legacy.get("line_interval_mm", None)
        )
        step.sample_interval_mm = data.get(
            "sample_interval_mm", legacy.get("sample_interval_mm", None)
        )
        step.dot_width_correction_mm = data.get(
            "dot_width_correction_mm", None
        )
        step.min_power_level = data.get(
            "min_power_level",
            legacy.get("min_power", data.get("min_power", 0.0)),
        )
        step.max_power_level = data.get(
            "max_power_level",
            legacy.get("max_power", data.get("max_power", 1.0)),
        )
        if "max_power_level" not in data:
            # Legacy engrave files stored the raster ceiling under the
            # max_power key; don't let it leak into the hardware max slot.
            step.max_power = 1000
        step.num_power_levels = int(
            data.get("num_power_levels", legacy.get("num_power_levels", 25))
        )
        step.offset_x_mm = data.get(
            "offset_x_mm", legacy.get("offset_x_mm", 0.0)
        )
        step.offset_y_mm = data.get(
            "offset_y_mm", legacy.get("offset_y_mm", 0.0)
        )
        scan_mode_str = data.get(
            "scan_mode", legacy.get("scan_mode", "SEGMENTED")
        )
        scan_mode_map = {
            "SEGMENTED": "SEGMENTED",
            "FULL_SWEEP": "FULL_SWEEP",
            "Segmented": "SEGMENTED",
            "FullSweep": "FULL_SWEEP",
        }
        step.scan_mode = scan_mode_map.get(scan_mode_str, "SEGMENTED")
        step.cross_hatch = data.get(
            "cross_hatch", legacy.get("cross_hatch", False)
        )
        step.num_depth_levels = int(
            data.get("num_depth_levels", legacy.get("num_depth_levels", 5))
        )
        step.z_step_down = data.get(
            "z_step_down", legacy.get("z_step_down", 0.0)
        )
        step.angle_increment = data.get(
            "angle_increment", legacy.get("angle_increment", 0.0)
        )
        dither_val = data.get(
            "dither_algorithm", legacy.get("dither_algorithm")
        )
        if dither_val is not None:
            try:
                step.dither_algorithm = DitherAlgorithm(dither_val)
            except ValueError:
                step.dither_algorithm = DitherAlgorithm.FLOYD_STEINBERG
        step.bidir_x_offset_mm = data.get("bidir_x_offset_mm", 0.0)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "scan_angle",
                "depth_mode",
                "invert",
                "auto_levels",
                "black_point",
                "white_point",
                "threshold",
                "line_interval_mm",
                "sample_interval_mm",
                "dot_width_correction_mm",
                "min_power_level",
                "max_power_level",
                "num_power_levels",
                "offset_x_mm",
                "offset_y_mm",
                "scan_mode",
                "cross_hatch",
                "num_depth_levels",
                "z_step_down",
                "angle_increment",
                "dither_algorithm",
                "bidir_x_offset_mm",
                "min_power",
                "max_power",
            }
        )

    @classmethod
    def get_default_transformers_dicts(cls) -> tuple[list, list]:
        OverscanTransformer = transformer_registry.get("OverscanTransformer")
        Optimize = transformer_registry.get("Optimize")
        MultiPassTransformer = transformer_registry.get("MultiPassTransformer")
        BidirScanOffsetTransformer = transformer_registry.get(
            "BidirScanOffsetTransformer"
        )
        assert OverscanTransformer is not None
        assert Optimize is not None
        assert MultiPassTransformer is not None
        assert BidirScanOffsetTransformer is not None
        optimize_dict = Optimize().to_dict()
        return [
            OverscanTransformer(
                enabled=True, distance_mm=0, auto=True
            ).to_dict(),
            optimize_dict,
            BidirScanOffsetTransformer(enabled=True).to_dict(),
        ], [
            optimize_dict,
            MultiPassTransformer(passes=1, z_step_down=0.0).to_dict(),
        ]

    @classmethod
    def create(
        cls,
        context: RayforgeContext,
        name: str | None = None,
        **kwargs,
    ) -> EngraveStep:
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
        # the operation's typical feed rate (engraving is faster than
        # cutting).
        step.cut_speed = min(machine.max_cut_speed, 4000)
        params = machine.get_pwm_params(default_head)
        if params is not None:
            step.frequency = params.frequency
            step.pulse_width = params.pulse_width

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

        return step


def _build_raster_part(
    step: EngraveStep,
    machine: Machine,
    workpiece: WorkPiece,
) -> tuple[Part, np.ndarray | None]:
    """Render and preprocess the workpiece into a :class:`Part`
    carrying a :class:`WholeImageSource`, and return the alpha
    channel separately so the caller can fold it into the
    :class:`RasterSpec`.

    The rendering resolution is clamped to
    :data:`MAX_RASTER_RENDER_PIXELS` to bound memory.  Auto-levels
    are precomputed here (see target-architecture.md B3.3) so all
    slabs see consistent black/white points.
    """
    size = workpiece.size
    if size[0] <= 0 or size[1] <= 0:
        return Part(size_mm=size), None

    spot_x, spot_y = LaserHead.get_spot_size(step.get_selected_laser(machine))
    px_per_mm_x = 1.0 / (step.sample_interval_mm or spot_x / 2.0)
    px_per_mm_y = 1.0 / spot_y

    target_w = max(1, int(size[0] * px_per_mm_x))
    target_h = max(1, int(size[1] * px_per_mm_y))
    num_pixels = target_w * target_h
    if num_pixels > MAX_RASTER_RENDER_PIXELS:
        scale = (MAX_RASTER_RENDER_PIXELS / num_pixels) ** 0.5
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))

    # Recompute pixels-per-mm from the actual integer target dimensions so
    # that the rendered image pixels exactly cover the workpiece size.
    # Without this, the int() truncation above leaves the image slightly
    # smaller than size_mm, shrinking the raster by up to one pixel.
    px_per_mm_x = target_w / size[0]
    px_per_mm_y = target_h / size[1]

    surface = workpiece.render_to_pixels(target_w, target_h)
    if surface is None:
        return Part(size_mm=size), None

    depth_mode = DepthMode[step.depth_mode]

    computed_auto_levels = None
    if step.auto_levels:
        computed_auto_levels = compute_raster_auto_levels(
            workpiece,
            (px_per_mm_x, px_per_mm_y),
            invert=step.invert,
        )

    image, alpha = preprocess_raster_image(
        surface,
        mode=depth_mode,
        invert=step.invert,
        auto_levels=step.auto_levels,
        computed_auto_levels=computed_auto_levels,
        black_point=step.black_point,
        white_point=step.white_point,
        threshold=step.threshold,
        dither_algorithm=step.dither_algorithm,
        laser_spot_x_mm=spot_x,
        pixels_per_mm_x=px_per_mm_x,
    )
    surface.flush()
    if image is None:
        return Part(size_mm=size), None

    part = Part(
        size_mm=size,
        pixels_per_mm=(px_per_mm_x, px_per_mm_y),
    )
    part.image_source = WholeImageSource(image)
    return part, alpha


MAX_RASTER_RENDER_PIXELS = 16 * 1024 * 1024
