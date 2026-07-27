from __future__ import annotations

from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    cast,
)

import numpy as np
from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops import Ops
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.raster import RasterSpec
from raygeo.ops.part import Part
from raygeo.ops.part.image_source import WholeImageSource
from raygeo.ops.types import SectionType

from rayforge.core.capability import ENGRAVE, Capability
from rayforge.core.step import Step
from rayforge.image.dither import DitherAlgorithm
from rayforge.pipeline.assembler.registry import assembler_registry
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    MachineDefaults,
    build_part_raster,
    compute_raster_auto_levels,
    make_artifact,
    preprocess_raster_image,
)
from rayforge.pipeline.transformer.registry import transformer_registry

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser
    from rayforge.pipeline.artifact import WorkPieceArtifact

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


class EngraveStep(Step):
    TYPELABEL = _("Engrave")
    ICON = "step-raster-symbolic"
    CAPABILITIES: Tuple[Capability, ...] = (ENGRAVE,)
    ASSEMBLER_NAME = "raster"
    IS_VECTOR = False
    ALWAYS_WRAP = True
    SECTION_TYPE = SectionType.RASTER_FILL

    def __init__(
        self, name: Optional[str] = None, typelabel: Optional[str] = None
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
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
        self.min_power = 0.0
        self.max_power = 1.0
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
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> dict:
        line_interval = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else machine_defaults.line_interval_mm
        )
        step_power = machine_defaults.step_power
        return {
            "mode": DepthMode[self.depth_mode].raygeo_name,
            "line_interval_mm": line_interval,
            "sample_interval_mm": self.sample_interval_mm,
            "dot_width_correction_mm": self.dot_width_correction_mm,
            "min_power": self.min_power,
            "max_power": self.max_power,
            "step_power": step_power,
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

    def build_compute_payload(
        self,
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> "Tuple[Part, ComputePayload]":
        """Build a :class:`Part` with the preprocessed raster image
        attached as a :class:`WholeImageSource`, and a
        :class:`ComputePayload` carrying a :class:`RasterSpec`.

        Rendering and preprocessing (dither / auto-levels / depth
        mode) happen here, on the calling thread, so the Rust
        assembler on the rayon worker only reads slabs from the
        attached image source.
        """
        part, alpha = _build_raster_part(self, machine_defaults, workpiece)
        kwargs = self.get_assembler_kwargs(machine_defaults, workpiece)
        depth_mode = DepthMode[self.depth_mode]
        spot_x = machine_defaults.tool_radius * 2.0
        line_interval = (
            kwargs["line_interval_mm"] or machine_defaults.line_interval_mm
        )
        sample_interval = kwargs["sample_interval_mm"] or spot_x
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
        machine_defaults: MachineDefaults,
        workpiece: "WorkPiece",
    ) -> Optional[dict]:
        return self.get_assembler_kwargs(machine_defaults, workpiece)

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
        result["min_power"] = self.min_power
        result["max_power"] = self.max_power
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
    def from_dict(cls, data: dict) -> "EngraveStep":
        step = cast("EngraveStep", super().from_dict(data))
        step.scan_angle = data.get("scan_angle", 0.0)
        step.depth_mode = data.get("depth_mode", "POWER_MODULATION")
        step.invert = data.get("invert", False)
        step.auto_levels = data.get("auto_levels", True)
        step.black_point = data.get("black_point", 0)
        step.white_point = data.get("white_point", 255)
        step.threshold = data.get("threshold", 128)
        step.line_interval_mm = data.get("line_interval_mm", None)
        step.sample_interval_mm = data.get("sample_interval_mm", None)
        step.dot_width_correction_mm = data.get(
            "dot_width_correction_mm", None
        )
        step.min_power = data.get("min_power", 0.0)
        step.max_power = data.get("max_power", 1.0)
        step.num_power_levels = data.get("num_power_levels", 25)
        step.offset_x_mm = data.get("offset_x_mm", 0.0)
        step.offset_y_mm = data.get("offset_y_mm", 0.0)
        step.scan_mode = data.get("scan_mode", "SEGMENTED")
        step.cross_hatch = data.get("cross_hatch", False)
        step.num_depth_levels = data.get("num_depth_levels", 5)
        step.z_step_down = data.get("z_step_down", 0.0)
        step.angle_increment = data.get("angle_increment", 0.0)
        dither_val = data.get("dither_algorithm")
        if dither_val is not None:
            step.dither_algorithm = DitherAlgorithm(dither_val)
        step.bidir_x_offset_mm = data.get("bidir_x_offset_mm", 0.0)
        return step

    def prepare(
        self,
        workpiece: "WorkPiece",
        settings: Dict[str, Any],
        resolved_params: Dict[str, Any],
    ) -> None:
        self._computed_auto_levels = None
        if not self.auto_levels:
            return
        self._computed_auto_levels = compute_raster_auto_levels(
            workpiece,
            settings["pixels_per_mm"],
            invert=self.invert,
        )

    def should_skip_workpiece(self, workpiece: "WorkPiece") -> bool:
        fills = workpiece.fills
        return fills is not None and len(fills) == 0

    def assemble_on_surface(
        self,
        workpiece: "WorkPiece",
        laser: "Laser",
        generation_id: int,
        surface: Any = None,
        pixels_per_mm: Optional[Tuple[float, float]] = None,
        *,
        machine_defaults: "MachineDefaults",
        y_offset_mm: float = 0.0,
        computed_auto_levels: Optional[Tuple[int, int]] = None,
    ) -> "WorkPieceArtifact":
        assert pixels_per_mm is not None
        assert surface is not None

        part = build_part_raster(workpiece, pixels_per_mm)
        rp = self.get_assembler_kwargs(machine_defaults, workpiece)

        width_px = surface.get_width()
        height_px = surface.get_height()

        depth_mode = DepthMode[self.depth_mode]

        if width_px == 0 or height_px == 0:
            final_ops = Ops()
            final_ops.ops_section_start(
                self.SECTION_TYPE,
                workpiece.uid,
                raster_mode=depth_mode.raster_mode,
            )
            final_ops.ops_section_end(
                self.SECTION_TYPE,
                raster_mode=depth_mode.raster_mode,
            )
            return make_artifact(
                final_ops,
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(0, 0),
            )

        image, alpha = preprocess_raster_image(
            surface,
            mode=depth_mode,
            invert=self.invert,
            auto_levels=self.auto_levels,
            computed_auto_levels=computed_auto_levels,
            black_point=self.black_point,
            white_point=self.white_point,
            threshold=self.threshold,
            dither_algorithm=self.dither_algorithm,
            laser_spot_x_mm=laser.spot_size_mm[0],
            pixels_per_mm_x=pixels_per_mm[0],
        )
        if image is None:
            return make_artifact(
                Ops(),
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(width_px, height_px),
            )
        part.image = image

        spot_y = laser.spot_size_mm[1]
        line_interval_mm = rp.get("line_interval_mm") or spot_y
        x_offset_mm = workpiece.bbox[0]
        y_off_mm = workpiece.bbox[1] + y_offset_mm
        sample_interval_mm = (
            rp.get("sample_interval_mm") or laser.spot_size_mm[0]
        )
        dot_width_correction_mm = (
            rp.get("dot_width_correction_mm")
            if rp.get("dot_width_correction_mm") is not None
            else laser.spot_size_mm[0] / 2.0
        )
        step_power = machine_defaults.step_power
        alpha_arr = (
            (alpha * 255).astype(np.uint8).tobytes()
            if alpha is not None
            else None
        )

        result = assembler_registry.assemble(
            self.ASSEMBLER_NAME,
            part,
            alpha=alpha_arr,
            mode=depth_mode.raygeo_name,
            line_interval_mm=line_interval_mm,
            sample_interval_mm=sample_interval_mm,
            min_power=rp.get("min_power", 0),
            max_power=rp.get("max_power", 100),
            step_power=step_power,
            num_power_levels=rp.get("num_power_levels", 256),
            angle=self.scan_angle,
            offset_x_mm=x_offset_mm,
            offset_y_mm=y_off_mm,
            scan_mode=rp.get("scan_mode", "segmented").lower(),
            cross_hatch=rp.get("cross_hatch", False),
            num_depth_levels=rp.get("num_depth_levels", 1),
            z_step_down=rp.get("z_step_down", 0.0),
            angle_increment=rp.get("angle_increment", 0),
            dot_width_correction_mm=dot_width_correction_mm,
        )

        final_ops = result.ops
        if final_ops.len() > 2:
            head_ops = Ops()
            head_ops.set_head(laser.uid)
            head_ops.extend(final_ops)
            final_ops = head_ops

        return make_artifact(
            final_ops,
            workpiece,
            generation_id,
            is_vector=self.IS_VECTOR,
            source_dimensions=(width_px, height_px),
        )

    @classmethod
    def get_default_transformers_dicts(cls) -> Tuple[List, List]:
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
        context: "RayforgeContext",
        name: Optional[str] = None,
        **kwargs,
    ) -> "EngraveStep":
        machine = context.machine
        assert machine is not None
        default_head = machine.get_default_head()

        step = cls(name=name)
        per_wp, per_step = cls.get_default_transformers_dicts()

        step.per_workpiece_transformers_dicts = per_wp
        step.per_step_transformers_dicts = per_step
        step.selected_laser_uid = default_head.uid
        step.max_cut_speed = machine.max_cut_speed
        step.max_travel_speed = machine.max_travel_speed
        for cap in machine.get_laser_capabilities(default_head):
            for var in cap.varset:
                setattr(step, var.key, var.default)

        # step.cut_speed is only final after the loop above.
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
    step: "EngraveStep",
    machine_defaults: MachineDefaults,
    workpiece: "WorkPiece",
) -> Tuple[Part, Optional[np.ndarray]]:
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

    spot_x = machine_defaults.tool_radius * 2.0
    spot_y = machine_defaults.line_interval_mm
    px_per_mm_x = 1.0 / (step.sample_interval_mm or spot_x)
    px_per_mm_y = 1.0 / spot_y

    target_w = int(size[0] * px_per_mm_x)
    target_h = int(size[1] * px_per_mm_y)
    num_pixels = target_w * target_h
    if num_pixels > MAX_RASTER_RENDER_PIXELS:
        scale = (MAX_RASTER_RENDER_PIXELS / num_pixels) ** 0.5
        target_w = max(1, int(target_w * scale))
        target_h = max(1, int(target_h * scale))
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
