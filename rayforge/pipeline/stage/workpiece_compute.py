import logging
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode
from raygeo.ops.types import SectionType

from ...core.step import Step
from ...core.workpiece import WorkPiece
from ...machine.models.laser import Laser
from ...shared.tasker.progress import ProgressContext, set_progress
from ..artifact import WorkPieceArtifact
from ..assembler.registry import assembler_registry
from ..transformer import ExecutionPhase, OpsTransformer
from .assembler_helpers import (
    DepthMode,
    MachineDefaults,
    build_part_raster,
    build_part_vector,
    compute_raster_auto_levels,
    make_artifact,
    preprocess_raster_image,
    resolve_machine_defaults,
    wrap_assembler_result,
)

if TYPE_CHECKING:
    from raygeo.geo import Geometry

logger = logging.getLogger(__name__)

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024
MAX_RASTER_RENDER_PIXELS = 16 * 1024 * 1024


def run_assembler_on_surface(
    step: "Step",
    workpiece: WorkPiece,
    laser: Laser,
    generation_id: int,
    surface: Any = None,
    pixels_per_mm: Optional[Tuple[float, float]] = None,
    *,
    machine_defaults: MachineDefaults,
    y_offset_mm: float = 0.0,
    computed_auto_levels: Optional[Tuple[int, int]] = None,
) -> WorkPieceArtifact:
    """Run an assembler on a surface (or vector data) and return an artifact.

    This replaces per-producer ``run()`` methods with a single generic
    dispatch driven by :class:`AssemblerMeta` orchestration metadata.
    """
    rp = step.get_assembler_kwargs(machine_defaults, workpiece)
    meta = assembler_registry.get(step.ASSEMBLER_NAME)
    if meta is None:
        raise KeyError(
            f"No assembler registered under '{step.ASSEMBLER_NAME}'"
        )

    # --- Build Part ------------------------------------------------
    if meta.build_part_mode == "vector":
        use_surface = surface is not None and (
            rp.get("override_threshold", False)
            or not workpiece.boundaries
            or workpiece.boundaries.is_empty()
        )
        part = build_part_vector(
            workpiece,
            surface=surface if use_surface else None,
            override_threshold=rp.get("override_threshold", False),
            threshold=rp.get("threshold", 0.5),
            normalize_windings=meta.normalize_windings,
        )
    elif meta.build_part_mode == "raster":
        assert pixels_per_mm is not None
        part = build_part_raster(workpiece, pixels_per_mm)
    else:
        part = None

    # --- Surface preprocessing (shrinkwrap etc.) -------------------
    if meta.prepare_surface is not None and surface is not None:
        assert part is not None
        boolean_image = meta.prepare_surface(surface)

        if not np.any(boolean_image):
            section_type = getattr(SectionType, meta.section_type)
            return make_artifact(
                Ops(),
                workpiece,
                generation_id,
                is_vector=meta.is_vector,
            )
        part.image = boolean_image

    # --- Raster image preprocessing --------------------------------
    if meta.build_part_mode == "raster" and surface is not None:
        assert part is not None
        width_px = surface.get_width()
        height_px = surface.get_height()

        if width_px == 0 or height_px == 0:
            section_type = getattr(SectionType, meta.section_type)
            final_ops = Ops()
            final_ops.ops_section_start(section_type, workpiece.uid)
            final_ops.ops_section_end(section_type)
            return make_artifact(
                final_ops,
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(0, 0),
            )

        depth_mode_str = rp.get("depth_mode", "POWER_MODULATION")

        depth_mode = DepthMode[depth_mode_str]
        image, alpha = preprocess_raster_image(
            surface,
            mode=depth_mode,
            invert=rp.get("invert", False),
            auto_levels=rp.get("auto_levels", True),
            computed_auto_levels=computed_auto_levels,
            black_point=rp.get("black_point", 0),
            white_point=rp.get("white_point", 255),
            threshold=rp.get("threshold", 128),
            laser_spot_x_mm=laser.spot_size_mm[0],
            pixels_per_mm_x=pixels_per_mm[0] if pixels_per_mm else 1.0,
        )
        if image is None:
            section_type = getattr(SectionType, meta.section_type)
            return make_artifact(
                Ops(),
                workpiece,
                generation_id,
                is_vector=False,
                source_dimensions=(width_px, height_px),
            )
        part.image = image

        # Resolve raster-specific params
        spot_y = laser.spot_size_mm[1]
        line_interval_mm = rp.get("line_interval_mm") or spot_y
        x_offset_mm = workpiece.bbox[0]
        y_off_mm = workpiece.bbox[1] + y_offset_mm
        sample_interval_mm = (
            rp.get("sample_interval_mm") or laser.spot_size_mm[0]
        )
        step_power = machine_defaults.step_power
        alpha_arr = (
            (alpha * 255).astype(np.uint8) if alpha is not None else None
        )

        result = assembler_registry.assemble(
            step.ASSEMBLER_NAME,
            part,
            alpha=alpha_arr,
            mode=depth_mode.raygeo_name,
            line_interval_mm=line_interval_mm,
            sample_interval_mm=sample_interval_mm,
            min_power=rp.get("min_power", 0),
            max_power=rp.get("max_power", 100),
            step_power=step_power,
            num_power_levels=rp.get("num_power_levels", 256),
            angle=rp.get("scan_angle", 0),
            offset_x_mm=x_offset_mm,
            offset_y_mm=y_off_mm,
            scan_mode=rp.get("scan_mode", "segmented").lower(),
            cross_hatch=rp.get("cross_hatch", False),
            num_depth_levels=rp.get("num_depth_levels", 1),
            z_step_down=rp.get("z_step_down", 0.0),
            angle_increment=rp.get("angle_increment", 0),
        )

        section_type = getattr(SectionType, meta.section_type)
        return wrap_assembler_result(
            result,
            workpiece,
            laser,
            generation_id,
            section_type=section_type,
            is_vector=False,
            source_dimensions=(width_px, height_px),
            always_wrap=meta.always_wrap,
        )

    # --- Vector / none path (contour / frame / shrinkwrap / wavefront
    #     / material_test_grid) ----------------------------------------
    if meta.build_part_mode == "vector" and (
        part is None or not part.has_geometry()
    ):
        logger.warning(
            "Assembler '%s' for '%s': no geometry available",
            step.ASSEMBLER_NAME,
            workpiece.name,
        )
        return make_artifact(
            Ops(),
            workpiece,
            generation_id,
            is_vector=meta.is_vector,
        )

    vec_kwargs = step.get_assembler_kwargs(machine_defaults, workpiece)

    result = assembler_registry.assemble(
        step.ASSEMBLER_NAME,
        part,
        **vec_kwargs,
    )

    set_power = machine_defaults.step_power if meta.set_power else None

    return wrap_assembler_result(
        result,
        workpiece,
        laser,
        generation_id,
        split_contours=meta.split_contours,
        set_power=set_power,
        is_vector=meta.is_vector,
    )


def _run_producer_on_surface(
    surface: Optional[Any],
    render_pixels_per_mm: Optional[Tuple[float, float]],
    step: "Step",
    laser: Laser,
    workpiece: WorkPiece,
    settings: dict,
    generation_id: int,
    generation_size: Optional[Tuple[float, float]] = None,
    y_offset_mm: float = 0.0,
    context: Optional[ProgressContext] = None,
    computed_auto_levels: Optional[Tuple[int, int]] = None,
) -> WorkPieceArtifact:
    """
    Run an assembler on a surface or vector data.

    Args:
        surface: The cairo.ImageSurface to process, or None for direct
            vector paths.
        render_pixels_per_mm: The actual pixels per mm used for rendering,
            or None for direct vector paths.
        laser: The Laser model with machine parameters.
        workpiece: The WorkPiece to generate operations for.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm, or None to use
            workpiece size.
        y_offset_mm: The vertical offset in mm for the current chunk.
        context: Optional ProgressContext for progress reporting.
        computed_auto_levels: Pre-computed auto-levels for raster.

    Returns:
        A WorkPieceArtifact containing the generated operations.
    """
    machine_defaults = resolve_machine_defaults(laser, settings)
    artifact = run_assembler_on_surface(
        step,
        workpiece,
        laser,
        generation_id,
        surface=surface,
        pixels_per_mm=render_pixels_per_mm,
        machine_defaults=machine_defaults,
        y_offset_mm=y_offset_mm,
        computed_auto_levels=computed_auto_levels,
    )
    if generation_size is not None:
        artifact.generation_size = generation_size
    return artifact


def _create_initial_ops(settings: dict) -> Ops:
    """
    Create and configure the initial Ops object with settings from the Step.

    Args:
        settings: Dictionary of settings from the Step.

    Returns:
        Configured Ops object.
    """
    initial_ops = Ops()
    initial_ops.set_power(settings["power"])
    initial_ops.set_feed_rate(settings["cut_speed"])
    initial_ops.set_rapid_rate(settings["travel_speed"])
    initial_ops.set_air_assist(
        AirAssistMode.ON if settings["air_assist"] else AirAssistMode.OFF
    )
    if settings.get("frequency"):
        initial_ops.set_frequency(settings["frequency"])
    if settings.get("pulse_width"):
        initial_ops.set_pulse_width(settings["pulse_width"])
    return initial_ops


def _validate_workpiece_size(
    size_mm: Optional[Tuple[float, float]],
) -> bool:
    """
    Validate that workpiece size is positive and non-zero.

    Args:
        size_mm: The workpiece size in mm as (width, height).

    Returns:
        True if size is valid, False otherwise.
    """
    if not size_mm:
        return False
    if size_mm[0] <= 0 or size_mm[1] <= 0:
        return False
    return True


def _calculate_vector_render_size(
    size_mm: Tuple[float, float],
    px_per_mm_x: float,
    px_per_mm_y: float,
) -> Tuple[int, int]:
    """
    Calculate render dimensions for vector operations with scaling.

    Args:
        size_mm: The workpiece size in mm as (width, height).
        px_per_mm_x: Horizontal pixels per mm.
        px_per_mm_y: Vertical pixels per mm.

    Returns:
        Tuple of (width_px, height_px) with scaling applied if needed.
    """
    target_width = int(size_mm[0] * px_per_mm_x)
    target_height = int(size_mm[1] * px_per_mm_y)

    num_pixels = target_width * target_height
    if num_pixels > MAX_VECTOR_TRACE_PIXELS:
        scale_factor = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
        target_width = int(target_width * scale_factor)
        target_height = int(target_height * scale_factor)

    return target_width, target_height


def _process_vector_render_and_trace(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Render workpiece to bitmap and trace for vector operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact or None if processing failed.
    """
    size_mm = workpiece.size
    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    target_width, target_height = _calculate_vector_render_size(
        size_mm, px_per_mm_x, px_per_mm_y
    )

    surface = workpiece.render_to_pixels(target_width, target_height)
    if not surface:
        return None

    artifact = _run_producer_on_surface(
        surface,
        None,
        step,
        laser,
        workpiece,
        settings,
        generation_id=generation_id,
        generation_size=generation_size,
        context=context,
    )

    surface.flush()
    return artifact


def _execute_vector(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Iterator[Tuple[WorkPieceArtifact, float]]:
    """
    Handle Ops generation for scalable (vector) operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Yields:
        A single tuple containing the complete Artifact and a
        progress value of 1.0.
    """
    size_mm = workpiece.size

    if not _validate_workpiece_size(size_mm):
        return

    meta = assembler_registry.get(step.ASSEMBLER_NAME)
    requires_full_render = meta.requires_full_render if meta else False

    if workpiece._edited_boundaries is not None:
        if not workpiece._edited_boundaries.is_empty():
            artifact = _run_producer_on_surface(
                surface=None,
                render_pixels_per_mm=None,
                step=step,
                laser=laser,
                workpiece=workpiece,
                settings=settings,
                generation_id=generation_id,
                generation_size=generation_size,
                context=context,
            )
            if artifact:
                yield artifact, 1.0
        return

    if workpiece.boundaries and not requires_full_render:
        artifact = _run_producer_on_surface(
            surface=None,
            render_pixels_per_mm=None,
            step=step,
            laser=laser,
            workpiece=workpiece,
            settings=settings,
            generation_id=generation_id,
            generation_size=generation_size,
            context=context,
        )
        if artifact:
            yield artifact, 1.0
        return

    artifact = _process_vector_render_and_trace(
        workpiece,
        step,
        laser,
        settings,
        generation_id,
        generation_size,
        context,
    )
    if artifact:
        yield artifact, 1.0


def _process_raster_full_render(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Process raster operation with full render at once.

    Args:
        workpiece: The WorkPiece to generate operations for.
        step: The Step object.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact or None if processing failed.
    """
    size = workpiece.size
    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    target_width = int(size[0] * px_per_mm_x)
    target_height = int(size[1] * px_per_mm_y)

    num_pixels = target_width * target_height
    if num_pixels > MAX_RASTER_RENDER_PIXELS:
        scale_factor = (MAX_RASTER_RENDER_PIXELS / num_pixels) ** 0.5
        target_width = max(1, int(target_width * scale_factor))
        target_height = max(1, int(target_height * scale_factor))

    effective_ppm_x = target_width / size[0] if size[0] > 0 else 0.0
    effective_ppm_y = target_height / size[1] if size[1] > 0 else 0.0

    surface = workpiece.render_to_pixels(target_width, target_height)
    if not surface:
        return None

    artifact = _run_producer_on_surface(
        surface,
        (effective_ppm_x, effective_ppm_y),
        step,
        laser,
        workpiece,
        settings,
        generation_id=generation_id,
        generation_size=generation_size,
        context=context,
    )

    surface.flush()
    return artifact


def _process_raster_chunk(
    surface: Any,
    x_offset_px: float,
    y_offset_px: float,
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    total_height_px: float,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
    computed_auto_levels: Optional[Tuple[int, int]] = None,
) -> Tuple[WorkPieceArtifact, float]:
    """
    Process a single raster chunk and calculate progress.

    Args:
        surface: The rendered surface chunk.
        x_offset_px: Horizontal offset in pixels.
        y_offset_px: Vertical offset in pixels.
        workpiece: The WorkPiece to generate operations for.
        step: The Step object.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        total_height_px: Total height in pixels for progress calculation.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Returns:
        Tuple of (chunk_artifact, progress).
    """
    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    progress = 0.0
    if total_height_px > 0:
        processed_height_px = y_offset_px + surface.get_height()
        progress = min(1.0, processed_height_px / total_height_px)

    y_offset_from_top_mm = y_offset_px / px_per_mm_y

    chunk_artifact = _run_producer_on_surface(
        surface,
        (px_per_mm_x, px_per_mm_y),
        step,
        laser,
        workpiece,
        settings,
        generation_id=generation_id,
        generation_size=generation_size,
        y_offset_mm=y_offset_from_top_mm,
        context=context,
        computed_auto_levels=computed_auto_levels,
    )

    size = workpiece.size
    y_offset_mm = (
        size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
    ) / px_per_mm_y
    x_offset_mm = x_offset_px / px_per_mm_x
    chunk_artifact.ops.translate(x_offset_mm, y_offset_mm)

    chunk_artifact.generation_size = generation_size

    return chunk_artifact, progress


def _has_no_fills(workpiece: WorkPiece) -> bool:
    """Check if a workpiece has vector geometry but no fills.

    Returns True for geometry-provider-based workpieces (e.g. sketches)
    that contain only stroked lines with no filled regions. Such
    workpieces should be handled by the vector pipeline, not rasterized.
    Returns False for image workpieces and workpieces with fills.
    """
    fills = workpiece.fills
    return fills is not None and len(fills) == 0


def _execute_raster(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Iterator[Tuple[WorkPieceArtifact, float]]:
    """
    Handle Ops generation for non-scalable (raster) operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Yields:
        A tuple for each chunk: (chunk_artifact, progress).
    """
    size = workpiece.size
    rp = step.get_assembler_kwargs(
        resolve_machine_defaults(laser, settings), workpiece
    )

    if not _validate_workpiece_size(size):
        return

    if _has_no_fills(workpiece):
        return

    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    total_pixels = size[0] * px_per_mm_x * size[1] * px_per_mm_y
    if total_pixels > MAX_RASTER_RENDER_PIXELS:
        scale_factor = (MAX_RASTER_RENDER_PIXELS / total_pixels) ** 0.5
        px_per_mm_x *= scale_factor
        px_per_mm_y *= scale_factor
        raster_settings = dict(settings)
        raster_settings["pixels_per_mm"] = (px_per_mm_x, px_per_mm_y)
    else:
        raster_settings = settings

    meta = assembler_registry.get(step.ASSEMBLER_NAME)
    requires_full_render = meta.requires_full_render if meta else False

    if requires_full_render:
        artifact = _process_raster_full_render(
            workpiece,
            step,
            laser,
            raster_settings,
            generation_id,
            generation_size,
            context,
        )
        if artifact:
            yield artifact, 1.0
        return

    computed_auto_levels = None
    if rp.get("auto_levels", False):
        computed_auto_levels = compute_raster_auto_levels(
            workpiece,
            raster_settings["pixels_per_mm"],
            invert=rp.get("invert", False),
        )

    total_height_px = size[1] * px_per_mm_y

    chunk_iter = workpiece.render_chunk(
        px_per_mm_x,
        px_per_mm_y,
        max_memory_size=10 * 1024 * 1024,
    )

    for surface, (x_offset_px, y_offset_px) in chunk_iter:
        chunk_artifact, progress = _process_raster_chunk(
            surface,
            x_offset_px,
            y_offset_px,
            workpiece,
            step,
            laser,
            raster_settings,
            generation_id,
            total_height_px,
            generation_size,
            context,
            computed_auto_levels,
        )

        yield chunk_artifact, progress
        surface.flush()


def _apply_transformers(
    ops: Ops,
    transformers: list[OpsTransformer],
    workpiece: WorkPiece,
    execute_weight: float,
    transform_weight: float,
    context: Optional[ProgressContext] = None,
    stock_geometries: Optional[List["Geometry"]] = None,
    settings: Optional[dict] = None,
) -> None:
    """
    Apply enabled transformers to the operations in phases.

    Args:
        ops: The Ops object to transform.
        transformers: List of OpsTransformer objects to apply.
        workpiece: The WorkPiece being processed.
        execute_weight: Weight for execute phase in progress calculation.
        transform_weight: Weight for transform phase in progress calculation.
        context: Optional ProgressContext for progress reporting.
        stock_geometries: Optional list of stock boundary geometries.
        settings: Optional dictionary of step settings.
    """
    enabled_transformers = [t for t in transformers if t.enabled]
    if not enabled_transformers:
        return

    phase_order = (
        ExecutionPhase.GEOMETRY_REFINEMENT,
        ExecutionPhase.PATH_INTERRUPTION,
        ExecutionPhase.POST_PROCESSING,
    )
    transformers_by_phase = {phase: [] for phase in phase_order}
    for t in enabled_transformers:
        transformers_by_phase[t.execution_phase].append(t)

    processed_count = 0
    total_to_process = len(enabled_transformers)

    for phase in phase_order:
        for transformer in transformers_by_phase[phase]:
            base = (
                execute_weight
                + (processed_count / total_to_process) * transform_weight
            )
            span = transform_weight / total_to_process
            if context is not None:
                transformer_ctx = context.sub_context(
                    base_progress=base,
                    progress_range=span,
                    total=1.0,
                )
            else:
                transformer_ctx = None
            set_progress(
                context,
                base,
                _("Applying '{transformer}' on '{workpiece}'").format(
                    transformer=transformer.label, workpiece=workpiece.name
                ),
            )
            transformer.run(
                ops,
                workpiece=workpiece,
                context=transformer_ctx,
                stock_geometries=stock_geometries,
                settings=settings,
            )
            processed_count += 1


def _merge_artifact_ops(
    final_artifact: Optional[WorkPieceArtifact],
    chunk_artifact: WorkPieceArtifact,
    initial_ops: Ops,
) -> WorkPieceArtifact:
    """
    Merge chunk artifact ops into final artifact.

    Args:
        final_artifact: The accumulated artifact or None for first chunk.
        chunk_artifact: The new chunk artifact to merge.
        initial_ops: The initial Ops configuration.

    Returns:
        The merged artifact.
    """
    if final_artifact is None:
        final_artifact = chunk_artifact
        new_ops = initial_ops.copy()
        new_ops.extend(final_artifact.ops)
        final_artifact.ops = new_ops
    else:
        final_artifact.ops.extend(chunk_artifact.ops)

    return final_artifact


def compute_workpiece_artifact_vector(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Compute a WorkPieceArtifact using vector operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact or None if generation failed.
    """
    set_progress(
        context,
        0.0,
        _("Generating path for '{workpiece}'").format(
            workpiece=workpiece.name
        ),
    )

    initial_ops = _create_initial_ops(settings)
    final_artifact: Optional[WorkPieceArtifact] = None

    execute_weight = 0.20

    for chunk_artifact, execute_progress in _execute_vector(
        workpiece,
        step,
        laser,
        settings,
        generation_id,
        generation_size,
        context,
    ):
        set_progress(
            context,
            execute_progress * execute_weight,
            _("Generating path for '{workpiece}'").format(
                workpiece=workpiece.name
            ),
        )

        final_artifact = _merge_artifact_ops(
            final_artifact,
            chunk_artifact,
            initial_ops,
        )

    set_progress(
        context,
        execute_weight,
        _("Generating path for '{workpiece}'").format(
            workpiece=workpiece.name
        ),
    )

    return final_artifact


def compute_workpiece_artifact_raster(
    workpiece: WorkPiece,
    step: "Step",
    laser: Laser,
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    on_chunk: Optional[Callable[[WorkPieceArtifact], None]] = None,
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Compute a WorkPieceArtifact using raster operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_id: The generation ID for staleness checking.
        generation_size: The full generation size in mm.
        on_chunk: Optional callback for progressive chunk reporting.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact or None if generation failed.
    """
    set_progress(
        context,
        0.0,
        _("Generating path for '{workpiece}'").format(
            workpiece=workpiece.name
        ),
    )

    initial_ops = _create_initial_ops(settings)
    final_artifact: Optional[WorkPieceArtifact] = None

    execute_weight = 0.20

    for chunk_artifact, execute_progress in _execute_raster(
        workpiece,
        step,
        laser,
        settings,
        generation_id,
        generation_size,
        context,
    ):
        set_progress(
            context,
            execute_progress * execute_weight,
            _("Generating path for '{workpiece}'").format(
                workpiece=workpiece.name
            ),
        )

        if on_chunk:
            on_chunk(chunk_artifact)

        final_artifact = _merge_artifact_ops(
            final_artifact,
            chunk_artifact,
            initial_ops,
        )

    set_progress(
        context,
        execute_weight,
        _("Generating path for '{workpiece}'").format(
            workpiece=workpiece.name
        ),
    )

    return final_artifact


def compute_workpiece_artifact(
    workpiece: WorkPiece,
    laser: Laser,
    transformers: List[OpsTransformer],
    settings: dict,
    pixels_per_mm: Tuple[float, float],
    generation_size: Tuple[float, float],
    generation_id: int,
    on_chunk: Optional[Callable[[WorkPieceArtifact], None]] = None,
    context: Optional[ProgressContext] = None,
    stock_geometries: Optional[List["Geometry"]] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Compute a WorkPieceArtifact from a WorkPiece.

    This is the core logic function that generates operations for a single
    (Step, WorkPiece) pair. It orchestrates the pipeline of assemblers
    and transformers.

    Args:
        workpiece: The WorkPiece to generate operations for.
        laser: The Laser model with machine parameters.
        transformers: List of OpsTransformer objects to apply to results.
        settings: Dictionary of settings from the Step
            (``step.to_dict()`` plus machine-derived keys).
        pixels_per_mm: Tuple of (x, y) pixels per millimeter.
        generation_size: The size of the generation in mm.
        generation_id: Unique identifier for this generation.
        on_chunk: Optional callback for progressive chunk reporting.
        context: Optional ProgressContext for progress reporting.
        stock_geometries: Optional list of stock boundary geometries.

    Returns:
        A WorkPieceArtifact containing the generated operations and metadata,
        or None if no artifact could be produced.
    """
    step = Step.from_dict(settings)
    meta = assembler_registry.get(step.ASSEMBLER_NAME)
    is_vector = meta.is_vector if meta else False

    if is_vector:
        final_artifact = compute_workpiece_artifact_vector(
            workpiece,
            step,
            laser,
            settings,
            generation_id,
            generation_size,
            context,
        )
    else:
        final_artifact = compute_workpiece_artifact_raster(
            workpiece,
            step,
            laser,
            settings,
            generation_id,
            generation_size,
            on_chunk=on_chunk,
            context=context,
        )

    if final_artifact is None:
        return None

    execute_weight = 0.20
    transform_weight = 1.0 - execute_weight

    _apply_transformers(
        final_artifact.ops,
        transformers,
        workpiece,
        execute_weight,
        transform_weight,
        context,
        stock_geometries,
        settings,
    )

    logger = logging.getLogger(__name__)
    logger.debug(
        "compute_workpiece_artifact: Creating final_artifact_to_store: "
        f"final_artifact.source_dimensions="
        f"{final_artifact.source_dimensions}, "
        f"generation_size={generation_size}"
    )
    final_artifact_to_store = WorkPieceArtifact(
        ops=final_artifact.ops,
        is_scalable=final_artifact.is_scalable,
        source_coordinate_system=final_artifact.source_coordinate_system,
        source_dimensions=final_artifact.source_dimensions,
        generation_size=generation_size,
        generation_id=generation_id,
    )
    logger.debug(
        "compute_workpiece_artifact: Created final_artifact_to_store: "
        f"source_dimensions={final_artifact_to_store.source_dimensions}, "
        f"generation_size={final_artifact_to_store.generation_size}"
    )

    set_progress(
        context,
        1.0,
        _("Finalizing '{workpiece}'").format(workpiece=workpiece.name),
    )

    return final_artifact_to_store
