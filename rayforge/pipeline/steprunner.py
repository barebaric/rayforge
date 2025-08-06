from typing import Any, List, Tuple, Iterator
from ..shared.tasker.proxy import ExecutionContextProxy


MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024
DEBOUNCE_DELAY_MS = 250  # Delay in milliseconds for ops regeneration


# This top-level function contains the core logic for generating Ops.
# It is designed to be run in a separate process by the TaskManager.
def run_step_in_subprocess(
    proxy: ExecutionContextProxy,
    # Pass all required state. Assume these are pickleable.
    workpiece_dict: dict[str, Any],
    opsproducer_dict: dict[str, Any],
    modifiers_dict: List[dict],
    opstransformers_dict: List[dict],
    laser_dict: dict[str, Any],
    settings: dict,
    generation_id: int,
):
    import logging

    logger = logging.getLogger(
        "rayforge.models.step.run_step_in_subprocess"
    )
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
    def _trace_and_modify_surface(surface, scaler, y_offset_mm=0.0):
        """Applies modifiers and runs the OpsProducer on a surface."""
        for modifier in modifiers:
            modifier.run(surface)
        return opsproducer.run(laser, surface, scaler, y_offset_mm=y_offset_mm)

    def _execute_vector() -> Iterator[Tuple[Ops, Tuple[int, int], float]]:
        """
        Handles Ops generation for scalable (vector) operations.
        This is a synchronous version of the original async method.
        """
        size_mm = workpiece.get_current_size()

        if not size_mm or None in size_mm:
            logger.warning(
                f"Cannot generate vector ops for '{workpiece.name}' "
                "without a defined size. Skipping."
            )
            return

        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]
        target_width = int(size_mm[0] * px_per_mm_x)
        target_height = int(size_mm[1] * px_per_mm_y)

        # Cap resolution
        num_pixels = target_width * target_height
        if num_pixels > MAX_VECTOR_TRACE_PIXELS:
            scale_factor = (MAX_VECTOR_TRACE_PIXELS / num_pixels) ** 0.5
            target_width = int(target_width * scale_factor)
            target_height = int(target_height * scale_factor)

        # This is now a blocking call, which is fine in a subprocess.
        surface = workpiece.renderer.render_to_pixels(
            width=target_width, height=target_height
        )
        if not surface:
            return

        pixel_scaler = 1.0, 1.0
        geometry_ops = _trace_and_modify_surface(surface, pixel_scaler)
        yield geometry_ops, (surface.get_width(), surface.get_height()), 1.0
        surface.flush()

    def _execute_raster() -> Iterator[Tuple[Ops, None, float]]:
        """
        Handles Ops generation for non-scalable (raster) operations.
        This is a synchronous version of the original async method.
        """
        size = workpiece.get_current_size()

        if not size or None in size:
            logger.warning(
                f"Cannot generate raster ops for '{workpiece.name}' "
                "without a defined size. Skipping."
            )
            return

        total_height_px = size[1] * settings["pixels_per_mm"][1]
        px_per_mm_x, px_per_mm_y = settings["pixels_per_mm"]

        # This iterator is now synchronous.
        chunk_iter = workpiece.render_chunk(
            px_per_mm_x,
            px_per_mm_y,
            size=size,
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

            chunk_ops = _trace_and_modify_surface(
                surface, (px_per_mm_x, px_per_mm_y),
                y_offset_mm=y_offset_from_top_mm
            )

            y_offset_mm = (
                size[1] * px_per_mm_y - (surface.get_height() + y_offset_px)
            ) / px_per_mm_y
            x_offset_mm = x_offset_px / px_per_mm_x
            chunk_ops.translate(x_offset_mm, y_offset_mm)

            yield chunk_ops, None, progress
            surface.flush()

    def _create_initial_ops():
        """Creates and new_configsures the initial Ops object."""
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
    final_ops = _create_initial_ops()
    cached_pixel_size = None
    is_vector = opsproducer.can_scale()

    execute_weight = 0.20
    transform_weight = 1.0 - execute_weight

    # --- Path generation phase ---
    execute_ctx = proxy.sub_context(
        base_progress=0.0, progress_range=execute_weight
    )
    execute_iterator = (
        _execute_vector() if is_vector else _execute_raster()
    )

    for chunk, px_size, execute_progress in execute_iterator:
        execute_ctx.set_progress(execute_progress)
        if px_size:
            cached_pixel_size = px_size
        if chunk:
            # For raster ops, send chunks for responsive UI. For vector ops,
            # do not send the unscaled chunk, as it will cause a visual glitch.
            if not is_vector:
                proxy.send_event('ops_chunk', {
                    'chunk': chunk,
                    'generation_id': generation_id
                })
        final_ops += chunk

    # Ensure path generation is marked as 100% complete before continuing.
    execute_ctx.set_progress(1.0)

    # --- Transform phase ---
    enabled_transformers = [t for t in opstransformers if t.enabled]
    num_transformers = len(enabled_transformers)
    if num_transformers > 0:
        transform_context = proxy.sub_context(
            base_progress=execute_weight, progress_range=transform_weight
        )

        for i, transformer in enumerate(enabled_transformers):
            proxy.set_message(
                _("Applying '{transformer}' on '{workpiece}'").format(
                    transformer=transformer.label,
                    workpiece=workpiece.name,
                )
            )
            # Create a proxy for this transformer's slice of the progress bar
            transformer_run_proxy = transform_context.sub_context(
                base_progress=(i / num_transformers),
                progress_range=(1 / num_transformers),
            )
            # transformer.run now runs synchronously and may use the proxy
            # to report its own fine-grained progress.
            transformer.run(final_ops, context=transformer_run_proxy)

            # Ensure this step's progress is marked as 100% complete before
            # moving to the next one. This prevents progress from appearing
            # to jump or stall if a transformer doesn't report its own
            # completion.
            transformer_run_proxy.set_progress(1.0)

    if settings["air_assist"]:
        final_ops.add(DisableAirAssistCommand())

    proxy.set_message(
        _("Finalizing '{workpiece}'").format(workpiece=workpiece.name)
    )
    proxy.set_progress(1.0)

    # The final result is returned and sent back by the _process_target_wrapper
    return final_ops, cached_pixel_size
