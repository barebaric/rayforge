from typing import Any, List, Tuple, Iterator, Optional, Callable
import logging
from ...core.ops import Ops
from ...core.workpiece import WorkPiece
from ...machine.models.laser import Laser
from ...shared.tasker.progress import ProgressContext, set_progress
from ..artifact import WorkPieceArtifact
from ..producer import OpsProducer
from ..transformer import OpsTransformer, ExecutionPhase

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024


def _run_producer_on_surface(
    surface: Optional[Any],
    render_pixels_per_mm: Optional[Tuple[float, float]],
    opsproducer: OpsProducer,
    laser: Laser,
    workpiece: WorkPiece,
    settings: dict,
    generation_size: Optional[Tuple[float, float]] = None,
    y_offset_mm: float = 0.0,
    context: Optional[ProgressContext] = None,
) -> WorkPieceArtifact:
    """
    Run the OpsProducer on a surface or vector data.

    Args:
        surface: The cairo.ImageSurface to process, or None for direct
            vector paths.
        render_pixels_per_mm: The actual pixels per mm used for rendering,
            or None for direct vector paths.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        workpiece: The WorkPiece to generate operations for.
        settings: Dictionary of settings from the Step.
        generation_size: The full generation size in mm, or None to use
            workpiece size.
        y_offset_mm: The vertical offset in mm for the current chunk.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact containing the generated operations.
    """
    artifact = opsproducer.run(
        laser,
        surface,
        render_pixels_per_mm,
        workpiece=workpiece,
        settings=settings,
        y_offset_mm=y_offset_mm,
        context=context,
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
    initial_ops.set_cut_speed(settings["cut_speed"])
    initial_ops.set_travel_speed(settings["travel_speed"])
    initial_ops.enable_air_assist(settings["air_assist"])
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
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Render workpiece to bitmap and trace for vector operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
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
        opsproducer,
        laser,
        workpiece,
        settings,
        generation_size=generation_size,
        context=context,
    )

    surface.flush()
    return artifact


def _execute_vector(
    workpiece: WorkPiece,
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Iterator[Tuple[WorkPieceArtifact, float]]:
    """
    Handle Ops generation for scalable (vector) operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Yields:
        A single tuple containing the complete Artifact and a
        progress value of 1.0.
    """
    size_mm = workpiece.size

    if not _validate_workpiece_size(size_mm):
        return

    if workpiece.boundaries and not opsproducer.requires_full_render:
        artifact = _run_producer_on_surface(
            surface=None,
            render_pixels_per_mm=None,
            opsproducer=opsproducer,
            laser=laser,
            workpiece=workpiece,
            settings=settings,
            generation_size=generation_size,
            context=context,
        )
        if artifact:
            yield artifact, 1.0
        return

    artifact = _process_vector_render_and_trace(
        workpiece,
        opsproducer,
        laser,
        settings,
        generation_size,
        context,
    )
    if artifact:
        yield artifact, 1.0


def _process_raster_full_render(
    workpiece: WorkPiece,
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Process raster operation with full render at once.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact or None if processing failed.
    """
    size = workpiece.size
    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    target_width = int(size[0] * px_per_mm_x)
    target_height = int(size[1] * px_per_mm_y)
    surface = workpiece.render_to_pixels(target_width, target_height)
    if not surface:
        return None

    artifact = _run_producer_on_surface(
        surface,
        (px_per_mm_x, px_per_mm_y),
        opsproducer,
        laser,
        workpiece,
        settings,
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
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    total_height_px: float,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Tuple[WorkPieceArtifact, float]:
    """
    Process a single raster chunk and calculate progress.

    Args:
        surface: The rendered surface chunk.
        x_offset_px: Horizontal offset in pixels.
        y_offset_px: Vertical offset in pixels.
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
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
        opsproducer,
        laser,
        workpiece,
        settings,
        generation_size=generation_size,
        y_offset_mm=y_offset_from_top_mm,
        context=context,
    )

    size = workpiece.size
    y_offset_mm = (
        size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
    ) / px_per_mm_y
    x_offset_mm = x_offset_px / px_per_mm_x
    chunk_artifact.ops.translate(x_offset_mm, y_offset_mm)

    chunk_artifact.generation_size = generation_size

    return chunk_artifact, progress


def _execute_raster(
    workpiece: WorkPiece,
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Iterator[Tuple[WorkPieceArtifact, float]]:
    """
    Handle Ops generation for non-scalable (raster) operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
        generation_size: The full generation size in mm.
        context: Optional ProgressContext for progress reporting.

    Yields:
        A tuple for each chunk: (chunk_artifact, progress).
    """
    size = workpiece.size

    if not _validate_workpiece_size(size):
        return

    px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

    if opsproducer.requires_full_render:
        artifact = _process_raster_full_render(
            workpiece,
            opsproducer,
            laser,
            settings,
            generation_size,
            context,
        )
        if artifact:
            yield artifact, 1.0
        return

    opsproducer.prepare(workpiece, settings)

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
            opsproducer,
            laser,
            settings,
            total_height_px,
            generation_size,
            context,
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
            set_progress(
                context,
                execute_weight
                + (processed_count / total_to_process) * transform_weight,
                _("Applying '{transformer}' on '{workpiece}'").format(
                    transformer=transformer.label, workpiece=workpiece.name
                ),
            )
            transformer.run(ops, workpiece=workpiece, context=None)
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
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Compute a WorkPieceArtifact using vector operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
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
        opsproducer,
        laser,
        settings,
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
    opsproducer: OpsProducer,
    laser: Laser,
    settings: dict,
    generation_size: Tuple[float, float],
    on_chunk: Optional[Callable[[WorkPieceArtifact], None]] = None,
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Compute a WorkPieceArtifact using raster operations.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        settings: Dictionary of settings from the Step.
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
        opsproducer,
        laser,
        settings,
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
    opsproducer: OpsProducer,
    laser: Laser,
    transformers: List[OpsTransformer],
    settings: dict,
    pixels_per_mm: Tuple[float, float],
    generation_size: Tuple[float, float],
    on_chunk: Optional[Callable[[WorkPieceArtifact], None]] = None,
    context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceArtifact]:
    """
    Computes a WorkPieceArtifact from a WorkPiece using the given producer.

    This is the core logic function that generates operations for a single
    (Step, WorkPiece) pair. It orchestrates the pipeline of producers
    and transformers.

    Args:
        workpiece: The WorkPiece to generate operations for.
        opsproducer: The OpsProducer to use for generation.
        laser: The Laser model with machine parameters.
        transformers: List of OpsTransformer objects to apply to results.
        settings: Dictionary of settings from the Step.
        pixels_per_mm: Tuple of (x, y) pixels per millimeter.
        generation_size: The size of the generation in mm.
        on_chunk: Optional callback for progressive chunk reporting.
        context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceArtifact containing the generated operations and metadata,
        or None if no artifact could be produced.
    """
    is_vector = opsproducer.is_vector_producer()

    if is_vector:
        final_artifact = compute_workpiece_artifact_vector(
            workpiece,
            opsproducer,
            laser,
            settings,
            generation_size,
            context,
        )
    else:
        final_artifact = compute_workpiece_artifact_raster(
            workpiece,
            opsproducer,
            laser,
            settings,
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
    )

    if settings["air_assist"]:
        final_artifact.ops.disable_air_assist()

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
