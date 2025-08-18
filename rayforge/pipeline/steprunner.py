from typing import Any, List, Tuple, Iterator, Optional
from ..shared.tasker.proxy import ExecutionContextProxy


MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024


# This top-level function contains the core logic for generating Ops.
# It is designed to be run in a separate process by the TaskManager.
def run_step_in_subprocess(
    proxy: ExecutionContextProxy,
    workpiece_dict: dict[str, Any],
    opsproducer_dict: dict[str, Any],
    modifiers_dict: List[dict],
    opstransformers_dict: List[dict],
    laser_dict: dict[str, Any],
    settings: dict,
    generation_id: int,
):
    """
    The main entry point for generating operations for a single
    (Step, WorkPiece) pair in a background process.

    This function reconstructs the necessary data models from dictionaries,
    determines whether the operation is vector or raster, and orchestrates the
    pipeline of producers, modifiers, and transformers.

    For vector operations, it produces unscaled Ops in a "natural" or "pixel"
    coordinate space and returns them with their dimensions. This allows the
    calling OpsGenerator to cache the result and perform fast scaling on
    demand.

    For raster operations, it produces Ops in their final millimeter
    coordinates, often in chunks, as they are not scalable.
    """
    import logging

    logger = logging.getLogger("rayforge.models.step.run_step_in_subprocess")
    logger.setLevel(proxy.parent_log_level)
    logger.debug(f"Starting step execution with settings: {settings}")

    from .modifier import Modifier
    from .producer import OpsProducer
    from .transformer import OpsTransformer
    from ..core.workpiece import WorkPiece
    from ..machine.models.laser import Laser
    from ..core.ops import Ops, DisableAirAssistCommand

    logger.debug("Imports completed")

    modifiers = [Modifier.from_dict(m) for m in modifiers_dict]
    opsproducer = OpsProducer.from_dict(opsproducer_dict)
    opstransformers = [
        OpsTransformer.from_dict(m) for m in opstransformers_dict
    ]
    laser = Laser.from_dict(laser_dict)
    workpiece = WorkPiece.from_dict(workpiece_dict)

    # Helper functions
    def _trace_and_modify_surface(
        surface: Optional[Any],
        scaler: Optional[Tuple[float, float]],
        *,
        y_offset_mm: float = 0.0,
    ) -> Ops:
        """
        Applies image modifiers and runs the OpsProducer on a surface or
        vector data.

        This is a central part of the pipeline. It first runs all configured
        modifiers (e.g., ToGrayscale) on the input surface if it exists. Then,
        it calls the main OpsProducer (e.g., Potrace, Rasterizer) to generate
        the machine operations.

        Args:
            surface: The cairo.ImageSurface to process, or None for direct
                vector paths.
            scaler: A tuple (pixels_per_mm_x, pixels_per_mm_y) to scale the
                output to millimeters, or None to get output in pixel
                coordinates.
            y_offset_mm: The vertical offset in mm for the current chunk, used
                by raster operations.

        Returns:
            An Ops object containing the generated operations.
        """
        for modifier in modifiers:
            # Modifiers only work on pixel surfaces, so skip if None
            if surface:
                modifier.run(surface)
        # Pass the importer to the producer for the vector fast-path
        return opsproducer.run(
            laser,
            surface,
            scaler,
            workpiece=workpiece,
            y_offset_mm=y_offset_mm,
        )

    def _execute_vector() -> Iterator[Tuple[Ops, Tuple[float, float], float]]:
        """
        Handles Ops generation for scalable (vector) operations.

        This function has two main paths:
        1. True Vector: If the workpiece's importer provides vector data
           directly (e.g., from an SVG), it is processed without rasterization.
           The resulting Ops are in the vector's "natural" coordinate system.
        2. Render-and-Trace: If no direct vector data is available, the
           workpiece is rendered to a high-resolution bitmap, which is then
           traced by a producer like Potrace. The resulting Ops are in pixel
           coordinates.

        In both cases, this function yields the unscaled Ops and the dimensions
        of their coordinate system, allowing for efficient caching and scaling
        by the OpsGenerator.

        Yields:
            A single tuple containing the complete Ops, the dimensions
            (width, height) of the coordinate system, and a progress value
            of 1.0.
        """
        size_mm = workpiece.size

        if not size_mm or size_mm[0] <= 0 or size_mm[1] <= 0:
            logger.warning(
                f"Cannot generate vector ops for '{workpiece.source_file}' "
                "without a valid, positive size. Skipping."
            )
            return

        # Path 1: True vector source (e.g., SVG).
        if workpiece.source_ops:
            logger.debug(
                "Workpiece has source_ops. Using direct vector processing."
            )
            geometry_ops = _trace_and_modify_surface(surface=None, scaler=None)
            # Return the unscaled ops along with their "natural" dimensions.
            natural_size = workpiece.get_natural_size()
            natural_w, natural_h = natural_size if natural_size else (1.0, 1.0)

            yield geometry_ops, (natural_w, natural_h), 1.0
            return

        # Path 2: Vector source that needs to be rendered and traced.
        logger.debug("No direct vector ops. Falling back to render-and-trace.")
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]
        target_width = int(size_mm[0] * px_per_mm_x)
        target_height = int(size_mm[1] * px_per_mm_y)

        # Cap resolution to prevent excessive memory usage.
        num_pixels = target_width * target_height
        if num_pixels > MAX_VECTOR_TRACE_PIXELS:
            scale_factor = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)

        # This is a blocking call, which is fine in a subprocess.
        surface = workpiece.render_to_pixels(target_width, target_height)
        if not surface:
            return

        # The producer (e.g., PotraceProducer) will trace the bitmap.
        # By passing `scaler=None`, it is expected to return ops in PIXEL
        # coordinates with a Y-up convention.
        geometry_ops = _trace_and_modify_surface(surface, None)

        yield geometry_ops, (surface.get_width(), surface.get_height()), 1.0
        surface.flush()

    def _execute_raster() -> Iterator[Tuple[Ops, None, float]]:
        """
        Handles Ops generation for non-scalable (raster) operations.

        This function renders the workpiece in horizontal chunks to manage
        memory usage. For each chunk, it generates Ops that are already scaled
        to final millimeter coordinates and correctly positioned. These chunks
        are yielded progressively to provide UI feedback.

        Yields:
            A tuple for each chunk: (chunk_ops, None, progress). `chunk_ops`
            are in mm, the pixel size is `None` (as they are pre-scaled), and
            `progress` is a float from 0.0 to 1.0.
        """
        size = workpiece.size

        if not size or size[0] <= 0 or size[1] <= 0:
            logger.warning(
                f"Cannot generate raster ops for '{workpiece.source_file}' "
                "without a defined size. Skipping."
            )
            return

        total_height_px = size[1] * settings["pixels_per_mm"][1]
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

        # render_chunk is an iterator that yields surfaces for processing.
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

            # The Rasterizer producer returns Ops pre-scaled to millimeters.
            chunk_ops = _trace_and_modify_surface(
                surface,
                (px_per_mm_x, px_per_mm_y),
                y_offset_mm=y_offset_from_top_mm,
            )

            # The ops are generated at the origin, so translate them to the
            # correct position within the workpiece.
            y_offset_mm = (
                size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
            ) / px_per_mm_y
            x_offset_mm = x_offset_px / px_per_mm_x
            chunk_ops.translate(x_offset_mm, y_offset_mm)

            yield chunk_ops, None, progress
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
        _("Generating path for '{name}'").format(
            name=workpiece.source_file.name
        )
    )
    final_ops = _create_initial_ops()
    final_pixel_size = None
    is_vector = opsproducer.can_scale()

    execute_weight = 0.20
    transform_weight = 1.0 - execute_weight

    # --- Path generation phase ---
    execute_ctx = proxy.sub_context(
        base_progress=0.0, progress_range=execute_weight
    )
    execute_iterator = _execute_vector() if is_vector else _execute_raster()

    for chunk, chunk_pixel_size, execute_progress in execute_iterator:
        execute_ctx.set_progress(execute_progress)
        if chunk_pixel_size:
            final_pixel_size = chunk_pixel_size

        if chunk:
            # Send intermediate chunks for raster operations
            # to provide responsive UI feedback during long-running jobs.
            if not is_vector:
                proxy.send_event(
                    "ops_chunk",
                    {"chunk": chunk, "generation_id": generation_id},
                )
            final_ops += chunk

    # Ensure path generation is marked as 100% complete before continuing.
    execute_ctx.set_progress(1.0)

    # --- Transform phase ---
    enabled_transformers = [t for t in opstransformers if t.enabled]
    if enabled_transformers:
        transform_context = proxy.sub_context(
            base_progress=execute_weight, progress_range=transform_weight
        )

        for i, transformer in enumerate(enabled_transformers):
            proxy.set_message(
                _("Applying '{transformer}' on '{workpiece}'").format(
                    transformer=transformer.label,
                    workpiece=workpiece.source_file.name,
                )
            )
            # Create a proxy for this transformer's slice of the progress bar
            transformer_run_proxy = transform_context.sub_context(
                base_progress=(i / len(enabled_transformers)),
                progress_range=(1 / len(enabled_transformers)),
            )
            # transformer.run now runs synchronously and may use the proxy
            # to report its own fine-grained progress.
            transformer.run(final_ops, context=transformer_run_proxy)

            # Ensure this step's progress is marked as 100% complete before
            # moving to the next one.
            transformer_run_proxy.set_progress(1.0)

    if settings["air_assist"]:
        final_ops.add(DisableAirAssistCommand())

    proxy.set_message(
        _("Finalizing '{workpiece}'").format(
            workpiece=workpiece.source_file.name
        )
    )
    proxy.set_progress(1.0)

    # Return the Ops and the pixel size (if any). The OpsGenerator will use
    # this to scale the ops correctly. For rasters, pixel size is None, as
    # they are already in their final millimeter coordinate system.
    return final_ops, final_pixel_size
