"""
Defines the dedicated subprocess for high-fidelity time estimation.
"""

from typing import Any, Tuple, List, Dict
from ..shared.tasker.proxy import ExecutionContextProxy
from .artifact.store import ArtifactStore
from .artifact.handle import ArtifactHandle
from ..pipeline.transformer import OpsTransformer, transformer_by_name
import logging

logger = logging.getLogger("rayforge.pipeline.timerunner")


def run_time_estimation_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_handle_dict: Dict[str, Any],
    target_size_mm: Tuple[float, float],
    per_step_transformers_dicts: List[Dict[str, Any]],
    cut_speed: float,
    travel_speed: float,
    acceleration: float,
    generation_id: int,
):
    """
    Calculates the machining time for a given artifact at a specific scale.

    This function is designed to be run in a separate process. It retrieves
    the base Ops from shared memory, scales it, applies post-transformers,
    and runs the full time estimation algorithm.

    Args:
        proxy: The proxy for communicating with the main process.
        artifact_handle_dict: A dictionary representation of the ArtifactHandle
                              for the base geometry.
        target_size_mm: The final (width, height) in mm to scale the Ops to.
        per_step_transformers_dicts: A list of dictionaries defining the
                                      per-step transformers to apply.
        cut_speed: The machine's maximum cut speed in mm/min.
        travel_speed: The machine's maximum travel speed in mm/min.
        acceleration: The machine's acceleration in mm/s^2.
        generation_id: An ID to prevent stale results.

    Returns:
        A tuple of (float, int), containing the estimated time in seconds
        and the generation_id.
    """
    logger.setLevel(proxy.parent_log_level)
    logger.debug(
        f"Starting time estimation for handle "
        f"{artifact_handle_dict.get('shm_name')}"
    )

    # --- 1. Reconstruct lightweight objects from dictionaries ---
    handle = ArtifactHandle.from_dict(artifact_handle_dict)
    artifact = ArtifactStore.get(handle)
    if not artifact:
        logger.error("Could not retrieve artifact from shared memory.")
        return 0.0, generation_id

    # Make a deep copy to avoid modifying the cached object
    ops = artifact.ops.copy()

    # --- 2. Scale the Ops if necessary ---
    if artifact.is_scalable and artifact.source_dimensions:
        source_w, source_h = artifact.source_dimensions
        target_w, target_h = target_size_mm
        if source_w > 1e-9 and source_h > 1e-9:
            scale_x = target_w / source_w
            scale_y = target_h / source_h
            ops.scale(scale_x, scale_y)

    # --- 3. Apply per-step transformers ---
    transformers: List[OpsTransformer] = []
    for t_dict in per_step_transformers_dicts:
        if not t_dict.get("enabled", True):
            continue
        cls_name = t_dict.get("name")
        if cls_name and cls_name in transformer_by_name:
            cls = transformer_by_name[cls_name]
            transformers.append(cls.from_dict(t_dict))

    for transformer in transformers:
        # We don't have a sub-context here, as this is a short,
        # non-cancellable task
        transformer.run(ops)

    # --- 4. Run the final time estimation ---
    estimated_time = ops.estimate_time(
        default_cut_speed=cut_speed,
        default_travel_speed=travel_speed,
        acceleration=acceleration,
    )
    logger.debug(f"Time estimation complete: {estimated_time:.2f} seconds.")

    return estimated_time, generation_id
