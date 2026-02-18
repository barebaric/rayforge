import cairo
import numpy as np
import logging
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.ops import Ops, SectionType
from ...image.image_util import surface_to_grayscale, surface_to_binary
from ...image.dither import surface_to_dithered_array, DitherAlgorithm
from ...shared.tasker.progress import ProgressContext
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer
from .raster_util import (
    find_segments,
    convert_y_to_output,
    calculate_ymax_mm,
    find_mask_bounding_box,
    generate_scan_lines,
)

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

logger = logging.getLogger(__name__)


class DepthMode(Enum):
    POWER_MODULATION = auto()
    CONSTANT_POWER = auto()
    DITHER = auto()
    MULTI_PASS = auto()


class DepthEngraver(OpsProducer):
    """
    Generates depth-engraving paths from a grayscale surface.
    """

    def __init__(
        self,
        scan_angle: float = 0.0,
        depth_mode: DepthMode = DepthMode.POWER_MODULATION,
        threshold: int = 128,
        dither_algorithm: Optional[
            DitherAlgorithm
        ] = DitherAlgorithm.FLOYD_STEINBERG,
        cross_hatch: bool = False,
        speed: float = 3000.0,
        min_power: float = 0.0,
        max_power: float = 1.0,
        num_depth_levels: int = 5,
        z_step_down: float = 0.0,
        invert: bool = False,
        line_interval_mm: Optional[float] = None,
    ):
        self.scan_angle = scan_angle
        self.depth_mode = depth_mode
        self.threshold = threshold
        self.dither_algorithm = dither_algorithm
        self.cross_hatch = cross_hatch
        self.speed = speed
        self.min_power = min_power
        self.max_power = max_power
        self.num_depth_levels = num_depth_levels
        self.z_step_down = z_step_down
        self.invert = invert
        self.line_interval_mm = line_interval_mm

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("DepthEngraver requires a workpiece context.")
        if surface.get_format() != cairo.FORMAT_ARGB32:
            raise ValueError("Unsupported Cairo surface format")

        final_ops = Ops()
        final_ops.ops_section_start(SectionType.RASTER_FILL, workpiece.uid)

        width_px = surface.get_width()
        height_px = surface.get_height()

        line_interval_mm = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else laser.spot_size_mm[1]
        )
        x_offset_mm = workpiece.bbox[0]
        y_offset_mm = workpiece.bbox[1] + y_offset_mm

        if width_px > 0 and height_px > 0:
            gray_image, alpha = surface_to_grayscale(surface)

            if self.invert:
                alpha_mask = alpha > 0
                gray_image[alpha_mask] = 255 - gray_image[alpha_mask]

            step_power = settings.get("power", 1.0) if settings else 1.0

            angles = [self.scan_angle]
            if self.cross_hatch:
                angles.append(self.scan_angle + 90.0)

            for angle in angles:
                if self.depth_mode == DepthMode.POWER_MODULATION:
                    mode_ops = self._run_power_modulation(
                        gray_image,
                        alpha,
                        pixels_per_mm,
                        x_offset_mm,
                        y_offset_mm,
                        line_interval_mm,
                        step_power,
                        angle,
                    )
                elif self.depth_mode == DepthMode.CONSTANT_POWER:
                    mode_ops = self._run_constant_power(
                        surface,
                        pixels_per_mm,
                        x_offset_mm,
                        y_offset_mm,
                        line_interval_mm,
                        laser,
                        step_power,
                        angle,
                        use_dither=False,
                    )
                elif self.depth_mode == DepthMode.DITHER:
                    mode_ops = self._run_constant_power(
                        surface,
                        pixels_per_mm,
                        x_offset_mm,
                        y_offset_mm,
                        line_interval_mm,
                        laser,
                        step_power,
                        angle,
                        use_dither=True,
                    )
                else:
                    mode_ops = self._run_multi_pass(
                        gray_image,
                        pixels_per_mm,
                        x_offset_mm,
                        y_offset_mm,
                        line_interval_mm,
                        angle,
                    )

                if not mode_ops.is_empty():
                    final_ops.set_laser(laser.uid)
                    final_ops.extend(mode_ops)

        final_ops.ops_section_end(SectionType.RASTER_FILL)

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(width_px, height_px),
            generation_size=workpiece.size,
        )

    def _run_power_modulation(
        self,
        gray_image: np.ndarray,
        alpha: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        step_power: float = 1.0,
        angle: float = 0.0,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = gray_image.shape
        ymax_mm = calculate_ymax_mm((width_px, height_px), pixels_per_mm)
        px_per_mm_x, px_per_mm_y = pixels_per_mm

        bbox = find_mask_bounding_box(alpha)
        if bbox is None:
            return ops

        power_range = self.max_power - self.min_power

        for scan_line in generate_scan_lines(
            bbox=bbox,
            image_size=(width_px, height_px),
            pixels_per_mm=pixels_per_mm,
            line_interval_mm=line_interval_mm,
            direction_degrees=angle,
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
        ):
            if len(scan_line.pixels) == 0:
                continue

            gray_values = gray_image[
                scan_line.pixels[:, 1], scan_line.pixels[:, 0]
            ]
            alpha_values = alpha[
                scan_line.pixels[:, 1], scan_line.pixels[:, 0]
            ]
            power_fractions = (
                self.min_power + (1.0 - gray_values / 255.0) * power_range
            )
            power_fractions = power_fractions * step_power
            power_values = (power_fractions * 255).astype(np.uint8)
            power_values[alpha_values == 0] = 0

            if not np.any(power_values > 0):
                continue

            segments = find_segments(power_values)
            if len(segments) == 0:
                continue

            is_reversed = (scan_line.index % 2) != 0
            if is_reversed:
                segments = segments[::-1]

            for start_idx, end_idx in segments:
                if power_values[start_idx] == 0:
                    continue

                power_slice = power_values[start_idx:end_idx]

                if is_reversed:
                    seg_start_px = scan_line.pixels[end_idx - 1]
                    seg_end_px = scan_line.pixels[start_idx]
                    power_slice = power_slice[::-1]
                else:
                    seg_start_px = scan_line.pixels[start_idx]
                    seg_end_px = scan_line.pixels[end_idx - 1]

                start_mm = scan_line.pixel_to_mm(
                    seg_start_px[0], seg_start_px[1], pixels_per_mm
                )
                end_mm = scan_line.pixel_to_mm(
                    seg_end_px[0], seg_end_px[1], pixels_per_mm
                )

                final_start_y = float(
                    convert_y_to_output(start_mm[1], ymax_mm)
                )
                final_end_y = float(convert_y_to_output(end_mm[1], ymax_mm))

                ops.move_to(start_mm[0], final_start_y, 0.0)
                ops.scan_to(
                    end_mm[0], final_end_y, 0.0, bytearray(power_slice)
                )

        return ops

    def _run_constant_power(
        self,
        surface,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        laser: "Laser",
        step_power: float = 1.0,
        angle: float = 0.0,
        use_dither: bool = False,
    ) -> Ops:
        ops = Ops()
        width_px = surface.get_width()
        height_px = surface.get_height()
        ymax_mm = calculate_ymax_mm((width_px, height_px), pixels_per_mm)
        px_per_mm_x, px_per_mm_y = pixels_per_mm

        min_feature_px = max(
            1, int(round(laser.spot_size_mm[0] * pixels_per_mm[0]))
        )

        if use_dither:
            dither_algo = (
                self.dither_algorithm or DitherAlgorithm.FLOYD_STEINBERG
            )
            mask = surface_to_dithered_array(
                surface,
                dither_algo,
                invert=self.invert,
                min_feature_px=min_feature_px,
            )
        else:
            mask = surface_to_binary(
                surface, threshold=self.threshold, invert=self.invert
            )

        bbox = find_mask_bounding_box(mask)
        if bbox is None:
            return ops

        for scan_line in generate_scan_lines(
            bbox=bbox,
            image_size=(width_px, height_px),
            pixels_per_mm=pixels_per_mm,
            line_interval_mm=line_interval_mm,
            direction_degrees=angle,
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
        ):
            if len(scan_line.pixels) == 0:
                continue

            values = mask[scan_line.pixels[:, 1], scan_line.pixels[:, 0]]
            segments = find_segments(values)

            if len(segments) == 0:
                continue

            is_reversed = (scan_line.index % 2) != 0
            if is_reversed:
                segments = segments[::-1]

            for start_idx, end_idx in segments:
                if values[start_idx] == 0:
                    continue

                if is_reversed:
                    seg_start_px = scan_line.pixels[end_idx - 1]
                    seg_end_px = scan_line.pixels[start_idx]
                else:
                    seg_start_px = scan_line.pixels[start_idx]
                    seg_end_px = scan_line.pixels[end_idx - 1]

                start_mm = scan_line.pixel_to_mm(
                    seg_start_px[0], seg_start_px[1], pixels_per_mm
                )
                end_mm = scan_line.pixel_to_mm(
                    seg_end_px[0], seg_end_px[1], pixels_per_mm
                )

                segment_length_px = end_idx - start_idx
                power_values = bytearray(
                    [int(255 * step_power)] * segment_length_px
                )

                final_start_y = float(
                    convert_y_to_output(start_mm[1], ymax_mm)
                )
                final_end_y = float(convert_y_to_output(end_mm[1], ymax_mm))

                ops.move_to(start_mm[0], final_start_y, 0.0)
                ops.scan_to(end_mm[0], final_end_y, 0.0, power_values)

        return ops

    def _run_multi_pass(
        self,
        gray_image: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        angle: float = 0.0,
    ) -> Ops:
        ops = Ops()

        pass_map = np.ceil(
            ((255 - gray_image) / 255.0) * self.num_depth_levels
        ).astype(int)

        for pass_level in range(1, self.num_depth_levels + 1):
            mask = (pass_map >= pass_level).astype(np.uint8)
            if not np.any(mask):
                continue

            z_offset = -((pass_level - 1) * self.z_step_down)
            pass_ops = self._rasterize_mask(
                mask,
                pixels_per_mm,
                offset_x_mm,
                offset_y_mm,
                line_interval_mm,
                z_offset,
                angle,
            )
            ops.extend(pass_ops)

        return ops

    def _rasterize_mask(
        self,
        mask: np.ndarray,
        pixels_per_mm: tuple,
        offset_x_mm: float,
        offset_y_mm: float,
        line_interval_mm: float,
        z: float,
        angle: float = 0.0,
    ) -> Ops:
        ops = Ops()
        height_px, width_px = mask.shape
        ymax_mm = calculate_ymax_mm((width_px, height_px), pixels_per_mm)

        bbox = find_mask_bounding_box(mask)
        if bbox is None:
            return ops

        for scan_line in generate_scan_lines(
            bbox=bbox,
            image_size=(width_px, height_px),
            pixels_per_mm=pixels_per_mm,
            line_interval_mm=line_interval_mm,
            direction_degrees=angle,
            offset_x_mm=offset_x_mm,
            offset_y_mm=offset_y_mm,
        ):
            if len(scan_line.pixels) == 0:
                continue

            values = mask[scan_line.pixels[:, 1], scan_line.pixels[:, 0]]
            segments = find_segments(values)

            if len(segments) == 0:
                continue

            is_reversed = (scan_line.index % 2) != 0
            if is_reversed:
                segments = segments[::-1]

            for start_idx, end_idx in segments:
                if values[start_idx] == 0:
                    continue

                if is_reversed:
                    seg_start_px = scan_line.pixels[end_idx - 1]
                    seg_end_px = scan_line.pixels[start_idx]
                else:
                    seg_start_px = scan_line.pixels[start_idx]
                    seg_end_px = scan_line.pixels[end_idx - 1]

                start_mm = scan_line.pixel_to_mm(
                    seg_start_px[0], seg_start_px[1], pixels_per_mm
                )
                end_mm = scan_line.pixel_to_mm(
                    seg_end_px[0], seg_end_px[1], pixels_per_mm
                )

                final_start_y = float(
                    convert_y_to_output(start_mm[1], ymax_mm)
                )
                final_end_y = float(convert_y_to_output(end_mm[1], ymax_mm))

                ops.move_to(start_mm[0], final_start_y, z)
                ops.line_to(end_mm[0], final_end_y, z)

        return ops

    def is_vector_producer(self) -> bool:
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {
                "scan_angle": self.scan_angle,
                "depth_mode": self.depth_mode.name,
                "threshold": self.threshold,
                "dither_algorithm": (
                    self.dither_algorithm.value
                    if self.dither_algorithm
                    else None
                ),
                "cross_hatch": self.cross_hatch,
                "speed": self.speed,
                "min_power": self.min_power,
                "max_power": self.max_power,
                "num_depth_levels": self.num_depth_levels,
                "z_step_down": self.z_step_down,
                "invert": self.invert,
                "line_interval_mm": self.line_interval_mm,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepthEngraver":
        """
        Deserializes a dictionary into a DepthEngraver instance.
        """
        params_in = data.get("params", {})

        valid_params = {
            "scan_angle",
            "depth_mode",
            "threshold",
            "dither_algorithm",
            "cross_hatch",
            "speed",
            "min_power",
            "max_power",
            "num_depth_levels",
            "z_step_down",
            "invert",
            "line_interval_mm",
        }

        init_args = {k: v for k, v in params_in.items() if k in valid_params}

        depth_mode_str = init_args.get(
            "depth_mode", DepthMode.POWER_MODULATION.name
        )
        try:
            init_args["depth_mode"] = DepthMode[depth_mode_str]
        except KeyError:
            init_args["depth_mode"] = DepthMode.POWER_MODULATION

        dither_algorithm_str = init_args.get("dither_algorithm")
        if dither_algorithm_str is not None:
            try:
                init_args["dither_algorithm"] = DitherAlgorithm(
                    dither_algorithm_str
                )
            except ValueError:
                init_args["dither_algorithm"] = DitherAlgorithm.FLOYD_STEINBERG
        else:
            init_args["dither_algorithm"] = None

        return cls(**init_args)
