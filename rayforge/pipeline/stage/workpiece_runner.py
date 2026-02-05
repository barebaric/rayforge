from typing import Any, List, Tuple, Optional, TYPE_CHECKING
from ...context import get_context
from ...shared.tasker.progress import CallbackProgressContext
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact.store import ArtifactStore
from .workpiece_compute import compute_workpiece_artifact

if TYPE_CHECKING:
    import threading


def make_workpiece_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    workpiece_dict: dict[str, Any],
    opsproducer_dict: dict[str, Any],
    modifiers_dict: List[dict],
    per_workpiece_transformers_dicts: List[dict],
    laser_dict: dict[str, Any],
    settings: dict,
    generation_id: int,
    generation_size: Tuple[float, float],
    creator_tag: str,
    adoption_event: Optional["threading.Event"] = None,
) -> int:
    """
    The main entry point for generating operations for a single (Step,
    WorkPiece) pair in a background process.

    This function reconstructs the necessary data models from dictionaries and
    calls the compute function to generate the artifact. The final generated
    artifact (containing Ops and metadata) is serialized into a shared memory
    block using the `ArtifactStore`. This function then returns the
    generation_id to signal completion.

    Args:
        proxy: The execution context proxy for progress reporting and events.
        artifact_store: The shared memory artifact store for serialization.
        workpiece_dict: Dictionary representation of the WorkPiece.
        opsproducer_dict: Dictionary representation of the OpsProducer.
        modifiers_dict: List of dictionaries for Modifiers.
        per_workpiece_transformers_dicts: List of dictionaries for
            OpsTransformers.
        laser_dict: Dictionary representation of the Laser.
        settings: The dictionary of settings from the Step.
        generation_id: Unique identifier for this generation.
        generation_size: The size of the generation in mm.
        creator_tag: Tag for artifact tracking.
        adoption_event: Optional event to wait for main process adoption.

    Returns:
        The generation_id to signal completion.
    """
    import logging

    logger = logging.getLogger(
        "rayforge.pipeline.steprunner.run_step_in_subprocess"
    )
    logger.debug(f"Starting step execution with settings: {settings}")

    from ..modifier import Modifier
    from ..producer import OpsProducer
    from ..transformer import OpsTransformer
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser

    logger.debug("Imports completed")

    artifact_store = get_context().artifact_store
    modifiers = [Modifier.from_dict(m) for m in modifiers_dict]
    opsproducer = OpsProducer.from_dict(opsproducer_dict)
    opstransformers = [
        OpsTransformer.from_dict(m) for m in per_workpiece_transformers_dicts
    ]
    laser = Laser.from_dict(laser_dict)
    workpiece = WorkPiece.from_dict(workpiece_dict)

    context = CallbackProgressContext(
        is_cancelled_func=proxy.is_cancelled,
        progress_callback=proxy.set_progress,
        message_callback=proxy.set_message,
    )

    pixels_per_mm = settings["pixels_per_mm"]

    final_artifact = compute_workpiece_artifact(
        workpiece=workpiece,
        opsproducer=opsproducer,
        laser=laser,
        modifiers=modifiers,
        transformers=opstransformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        context=context,
    )

    if final_artifact is None:
        # If no artifact was produced (e.g., empty image), we still need
        # to return the generation_id to signal completion.
        return generation_id

    handle = artifact_store.put(final_artifact, creator_tag=creator_tag)
    proxy.send_event(
        "artifact_created",
        {"handle_dict": handle.to_dict(), "generation_id": generation_id},
    )

    # Wait for main process to adopt the artifact before forgetting it
    if adoption_event is not None:
        logger.debug("Waiting for main process to adopt workpiece artifact...")
        if adoption_event.wait(timeout=10):
            logger.debug(
                "Main process adopted workpiece artifact. Forgetting..."
            )
            artifact_store.forget(handle)
            logger.info("Worker disowned workpiece artifact successfully")
        else:
            logger.warning(
                "Main process failed to adopt workpiece artifact within "
                "timeout. Releasing to prevent leak."
            )
            artifact_store.release(handle)

    return generation_id
