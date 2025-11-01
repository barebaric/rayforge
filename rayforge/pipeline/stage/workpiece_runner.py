from typing import Any, List, Tuple, Iterator, Optional, Dict
import numpy as np
from ...shared.tasker.proxy import ExecutionContextProxy
from ..encoder.vertexencoder import VertexEncoder
from ...context import get_context
from ..artifact import WorkPieceArtifact

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024


# This top-level function contains the core logic for generating Ops.
# It is designed to be run in a separate process by the TaskManager.
def make_workpiece_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    workpiece_dict: dict[str, Any],
    opsproducer_dict: dict[str, Any],
    modifiers_dict: List[dict],
    per_workpiece_transformers_dicts: List[dict],
    laser_dict: dict[str, Any],
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    creator_tag: str,
) -> int:
    """
    The main entry point for generating operations for a single (Step,
    WorkPiece) pair in a background process.

    This function reconstructs the necessary data models from dictionaries and
    orchestrates the pipeline of producers, modifiers, and transformers.

    The final generated artifact (containing Ops and metadata) is serialized
    into a shared memory block using the `ArtifactStore`. This function then
    returns the generation_id to signal completion.
    """
    import logging

    logger = logging.getLogger(
        "rayforge.pipeline.steprunner.run_step_in_subprocess"
    )
    logger.debug(f"Starting step execution with settings: {settings}")

    from ..modifier import Modifier
    from ..producer import OpsProducer
    from ..transformer import OpsTransformer, ExecutionPhase
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...core.ops import Ops

    logger.debug("Imports completed")

    artifact_store = get_context().artifact_store
    modifiers = [Modifier.from_dict(m) for m in modifiers_dict]
    opsproducer = OpsProducer.from_dict(opsproducer_dict)
    opstransformers = [
        OpsTransformer.from_dict(m) for m in per_workpiece_transformers_dicts
    ]
    laser = Laser.from_dict(laser_dict)
    workpiece = WorkPiece.from_dict(workpiece_dict)

    # Helper functions
    def _trace_and_modify_surface(
        surface: Optional[Any],
        render_pixels_per_mm: Optional[Tuple[float, float]],
        *,
        y_offset_mm: float = 0.0,
        step_settings: Dict[str, Any],
    ) -> WorkPieceArtifact:
        """
        Applies image modifiers and runs the OpsProducer on a surface or
        vector data.

        This is a central part of the pipeline. It first runs all configured
        modifiers (e.g., ToGrayscale) on the input surface if it exists. Then,
        it calls the main OpsProducer (e.g., Rasterizer) to generate
        the machine operations.

        Args:
            surface: The cairo.ImageSurface to process, or None for direct
                vector paths.
            render_pixels_per_mm: The actual pixels per mm used for rendering,
                or None for direct vector paths.
            y_offset_mm: The vertical offset in mm for the current chunk, used
                by raster operations.
            step_settings: The dictionary of settings from the Step.

        Returns:
            A Artifact object containing the generated operations
              and metadata.
        """
        for modifier in modifiers:
            # Modifiers only work on pixel surfaces, so skip if None
            if surface:
                modifier.run(surface)
        # Pass the importer to the producer for the vector fast-path
        return opsproducer.run(
            laser,
            surface,
            render_pixels_per_mm,
            workpiece=workpiece,
            settings=step_settings,
            y_offset_mm=y_offset_mm,
        )

    def _execute_vector() -> Iterator[Tuple[WorkPieceArtifact, float]]:
        """
        Handles Ops generation for scalable (vector) operations.

        This function has two main paths:
        1. True Vector: If the workpiece's importer provides vector data
           directly (e.g., from an SVG), it is processed without rasterization.
           The resulting Ops are in the vector's "natural" coordinate system.
        2. Render-and-Trace: If no direct vector data is available, or if the
           producer requires it, the workpiece is rendered to a bitmap, which
           is then traced. The resulting Ops are in pixel coordinates.

        In both cases, this function yields the unscaled Ops and the dimensions
        of their coordinate system, allowing for efficient caching and scaling
        by the Pipeline.

        Yields:
            A single tuple containing the complete Artifact and a
            progress value of 1.0.
        """
        size_mm = workpiece.size

        if not size_mm or size_mm[0] <= 0 or size_mm[1] <= 0:
            logger.warning(
                f"Cannot generate vector ops for '{workpiece.name}' "
                "without a valid, positive size. Skipping."
            )
            return

        # Path 1: True vector source (e.g., SVG).
        if workpiece.vectors and not opsproducer.requires_full_render:
            logger.debug(
                "Workpiece has vectors and producer does not require a full "
                "render. Using direct vector processing."
            )
            artifact = _trace_and_modify_surface(
                surface=None,
                render_pixels_per_mm=None,
                step_settings=settings,
            )
            yield artifact, 1.0
            return

        # Path 2: Render-and-trace.
        if opsproducer.requires_full_render:
            logger.debug(
                "Producer requires a full render. Forcing render-and-trace."
            )
        else:
            logger.debug(
                "No direct vector ops. Falling back to render-and-trace."
            )
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]
        target_width = int(size_mm[0] * px_per_mm_x)
        target_height = int(size_mm[1] * px_per_mm_y)

        # Cap resolution to prevent excessive memory usage.
        num_pixels = target_width * target_height
        if num_pixels > MAX_VECTOR_TRACE_PIXELS:
            scale_factor = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)

        logger.debug(
            f"Vector producer rendering '{workpiece.name}' to "
            f"{target_width}x{target_height} px."
        )
        # This is a blocking call, which is fine in a subprocess.
        surface = workpiece.render_to_pixels(target_width, target_height)
        if not surface:
            return

        # The producer (e.g., EdgeTracer) will trace the bitmap and return
        # an artifact with pixel coordinates.
        artifact = _trace_and_modify_surface(
            surface, None, step_settings=settings
        )

        yield artifact, 1.0
        surface.flush()

    def _execute_raster() -> Iterator[Tuple[WorkPieceArtifact, float]]:
        """
        Handles Ops generation for non-scalable (raster) operations.

        This function renders the workpiece in horizontal chunks to manage
        memory usage. For each chunk, it generates an artifact containing Ops
        that are already scaled to final millimeter coordinates. These chunks
        are yielded progressively to provide UI feedback.

        Yields:
            A tuple for each chunk: (chunk_artifact, progress).
        """
        size = workpiece.size

        if not size or size[0] <= 0 or size[1] <= 0:
            logger.warning(
                f"Cannot generate raster ops for '{workpiece.name}' "
                "without a defined size. Skipping."
            )
            return

        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

        # Special case for scalable producers that need a full render.
        if opsproducer.requires_full_render:
            logger.debug("Producer requires full render, bypassing chunking.")
            target_width = int(size[0] * px_per_mm_x)
            target_height = int(size[1] * px_per_mm_y)
            surface = workpiece.render_to_pixels(target_width, target_height)
            if not surface:
                return

            full_artifact = _trace_and_modify_surface(
                surface,
                (px_per_mm_x, px_per_mm_y),
                step_settings=settings,
            )
            yield full_artifact, 1.0
            surface.flush()
            return

        # Default chunking behavior
        total_height_px = size[1] * px_per_mm_y

        chunk_iter = workpiece.render_chunk(
            px_per_mm_x,
            px_per_mm_y,
            max_memory_size=10 * 1024 * 1024,
        )

        for surface, (x_offset_px, y_offset_px) in chunk_iter:
            progress = 0.0
            if total_height_px > 0:
                processed_height_px = y_offset_px + surface.get_height()
                progress = min(1.0, processed_height_px / total_height_px)

            # Calculate the absolute Y offset of this chunk from the top of
            # the workpiece. This is crucial for aligning raster lines across
            # chunks.
            y_offset_from_top_mm = y_offset_px / px_per_mm_y

            # The Rasterizer producer returns an artifact with Ops pre-scaled
            # to millimeters.
            chunk_artifact = _trace_and_modify_surface(
                surface,
                (px_per_mm_x, px_per_mm_y),
                y_offset_mm=y_offset_from_top_mm,
                step_settings=settings,
            )

            if chunk_artifact.texture_data:
                chunk_artifact.texture_data.power_texture_data = (
                    chunk_artifact.texture_data.power_texture_data.copy()
                )

            y_offset_mm = (
                size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
            ) / px_per_mm_y
            x_offset_mm = x_offset_px / px_per_mm_x
            chunk_artifact.ops.translate(x_offset_mm, y_offset_mm)

            yield chunk_artifact, progress
            surface.flush()

    def _create_initial_ops() -> Ops:
        """
        Creates and configures the initial Ops object with settings from the
        Step.
        """
        initial_ops = Ops()
        initial_ops.set_power(settings["power"])
        initial_ops.set_cut_speed(settings["cut_speed"])
        initial_ops.set_travel_speed(settings["travel_speed"])
        initial_ops.enable_air_assist(settings["air_assist"])
        return initial_ops

    # === Main execution logic for the subprocess ===

    proxy.set_message(
        _("Generating path for '{name}'").format(name=workpiece.name)
    )
    initial_ops = _create_initial_ops()
    chunk_artifacts_for_assembly = []

    is_vector = opsproducer.is_vector_producer()
    encoder = VertexEncoder()

    execute_weight = 0.20
    transform_weight = 1.0 - execute_weight

    # Path generation phase
    execute_ctx = proxy.sub_context(
        base_progress=0.0, progress_range=execute_weight
    )
    execute_iterator = _execute_vector() if is_vector else _execute_raster()

    for chunk_artifact, execute_progress in execute_iterator:
        execute_ctx.set_progress(execute_progress)
        chunk_artifacts_for_assembly.append(chunk_artifact)

        if not is_vector:
            import copy

            # For progressive rendering, we create a separate artifact copy.
            # This prevents mutation of the original chunk_artifact which is
            # used to assemble the final texture.
            chunk_for_view = copy.deepcopy(chunk_artifact)

            # The producer sets texture position relative to the top-left
            # (Y-down). We must convert it to bottom-left (Y-up) for the
            # view renderer, which uses a consistent Y-up coordinate system
            # for both ops and textures.
            if chunk_for_view.texture_data:
                size_mm = workpiece.size
                (
                    tex_pos_x,
                    tex_pos_y_from_top,
                ) = chunk_for_view.texture_data.position_mm
                _tex_w, tex_h = chunk_for_view.texture_data.dimensions_mm
                total_h = size_mm[1]

                # New Y is calculated from the bottom of the workpiece.
                new_y_pos_from_bottom = total_h - tex_pos_y_from_top - tex_h
                chunk_for_view.texture_data.position_mm = (
                    tex_pos_x,
                    new_y_pos_from_bottom,
                )

            # Encode vertices for the current chunk.
            ops_for_chunk_render = initial_ops.copy()
            ops_for_chunk_render.extend(chunk_for_view.ops)
            chunk_for_view.vertex_data = encoder.encode(ops_for_chunk_render)

            # Store in shared memory and get a handle
            chunk_handle = artifact_store.put(
                chunk_for_view, creator_tag=f"{creator_tag}_chunk"
            )
            proxy.send_event(
                "visual_chunk_ready",
                {
                    "handle_dict": chunk_handle.to_dict(),
                    "generation_id": generation_id,
                },
            )

    # Ensure path generation is marked as 100% complete before continuing.
    execute_ctx.set_progress(1.0)

    if not chunk_artifacts_for_assembly:
        # If no artifact was produced (e.g., empty image), we still need
        # to return the generation_id to signal completion.
        return generation_id

    # Assemble final artifact from all chunks
    final_artifact = chunk_artifacts_for_assembly[0]
    final_ops = initial_ops.copy()
    final_ops.extend(final_artifact.ops)

    for i in range(1, len(chunk_artifacts_for_assembly)):
        final_ops.extend(chunk_artifacts_for_assembly[i].ops)
    final_artifact.ops = final_ops

    if not is_vector and final_artifact.texture_data:
        size_mm = workpiece.size
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]
        full_width_px = int(size_mm[0] * px_per_mm_x)
        full_height_px = int(size_mm[1] * px_per_mm_y)
        if full_width_px > 0 and full_height_px > 0:
            full_power_texture = np.zeros(
                (full_height_px, full_width_px), dtype=np.uint8
            )
            for chunk in chunk_artifacts_for_assembly:
                if chunk.texture_data:
                    texture_data = chunk.texture_data.power_texture_data
                    chunk_h_px, chunk_w_px = texture_data.shape
                    y_start_px = int(
                        round(chunk.texture_data.position_mm[1] * px_per_mm_y)
                    )
                    x_start_px = 0
                    y_end_px = y_start_px + chunk_h_px
                    x_end_px = x_start_px + chunk_w_px
                    if (
                        y_end_px <= full_power_texture.shape[0]
                        and x_end_px <= full_power_texture.shape[1]
                    ):
                        full_power_texture[
                            y_start_px:y_end_px, x_start_px:x_end_px
                        ] = texture_data
                    else:
                        logger.warning("Chunk texture out of bounds.")

            final_artifact.texture_data.power_texture_data = full_power_texture
            final_artifact.texture_data.position_mm = (0.0, 0.0)
            final_artifact.texture_data.dimensions_mm = workpiece.size
            final_artifact.source_dimensions = (
                full_power_texture.shape[1],
                full_power_texture.shape[0],
            )

    # Transform phase
    enabled_transformers = [t for t in opstransformers if t.enabled]
    if enabled_transformers:
        transform_context = proxy.sub_context(
            base_progress=execute_weight, progress_range=transform_weight
        )

        # 1. Group transformers by their execution phase
        phase_order = (
            ExecutionPhase.GEOMETRY_REFINEMENT,
            ExecutionPhase.PATH_INTERRUPTION,
            ExecutionPhase.POST_PROCESSING,
        )
        transformers_by_phase = {phase: [] for phase in phase_order}
        for t in enabled_transformers:
            transformers_by_phase[t.execution_phase].append(t)

        # 2. Execute transformers in the correct phase order
        processed_count = 0
        total_to_process = len(enabled_transformers)

        for phase in phase_order:
            for transformer in transformers_by_phase[phase]:
                proxy.set_message(
                    _("Applying '{transformer}' on '{workpiece}'").format(
                        transformer=transformer.label,
                        workpiece=workpiece.name,
                    )
                )
                # Create a proxy for this transformer's slice of progress
                transformer_run_proxy = transform_context.sub_context(
                    base_progress=(processed_count / total_to_process),
                    progress_range=(1 / total_to_process),
                )
                # transformer.run now runs synchronously and may use the
                # proxy to report its own fine-grained progress.
                transformer.run(
                    final_artifact.ops,
                    workpiece=workpiece,
                    context=transformer_run_proxy,
                )

                # Mark step as complete and increment for the next one
                transformer_run_proxy.set_progress(1.0)
                processed_count += 1

    if settings["air_assist"]:
        final_artifact.ops.disable_air_assist()

    # After all transformations, encode the final Ops into vertex data and
    # create the final artifact for storage.
    logger.debug("Encoding final ops into vertex data for rendering.")
    vertex_data = encoder.encode(final_artifact.ops)

    final_artifact_to_store = WorkPieceArtifact(
        ops=final_artifact.ops,
        is_scalable=final_artifact.is_scalable,
        source_coordinate_system=final_artifact.source_coordinate_system,
        source_dimensions=final_artifact.source_dimensions,
        generation_size=generation_size,
        vertex_data=vertex_data,
        texture_data=final_artifact.texture_data,
    )

    proxy.set_message(
        _("Finalizing '{workpiece}'").format(workpiece=workpiece.name)
    )
    proxy.set_progress(1.0)

    handle = artifact_store.put(
        final_artifact_to_store, creator_tag=creator_tag
    )
    proxy.send_event(
        "artifact_created",
        {"handle_dict": handle.to_dict(), "generation_id": generation_id},
    )

    return generation_id
