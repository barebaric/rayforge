from typing import Any, List, Tuple, TYPE_CHECKING
from ...shared.tasker.progress import CallbackProgressContext
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact.store import ArtifactStore
from .workpiece_compute import compute_workpiece_artifact

if TYPE_CHECKING:
    from ..artifact import WorkPieceArtifact


def make_workpiece_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    workpiece_dict: dict[str, Any],
    opsproducer_dict: dict[str, Any],
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
    calls the compute function to generate the artifact. The final generated
    artifact (containing Ops and metadata) is serialized into a shared memory
    block using the `ArtifactStore`.

    This implementation uses a "Fire and Forget" strategy for artifact
    handover. The worker stores the artifact, sends the handle to the main
    process, and immediately "forgets" (closes) its reference. This prevents
    deadlocks caused by waiting for the main process to adopt.

    Args:
        proxy: The execution context proxy for progress reporting and events.
        artifact_store: The shared memory artifact store for serialization.
        workpiece_dict: Dictionary representation of the WorkPiece.
        opsproducer_dict: Dictionary representation of the OpsProducer.
        per_workpiece_transformers_dicts: List of dictionaries for
            OpsTransformers.
        laser_dict: Dictionary representation of the Laser.
        settings: The dictionary of settings from the Step.
        generation_id: Unique identifier for this generation.
        generation_size: The size of the generation in mm.
        creator_tag: Tag for artifact tracking.

    Returns:
        The generation_id to signal completion.
    """
    import logging

    logger = logging.getLogger(
        "rayforge.pipeline.steprunner.run_step_in_subprocess"
    )
    logger.debug(f"Starting step execution with settings: {settings}")

    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ..producer import OpsProducer
    from ..transformer import OpsTransformer

    logger.debug("Imports completed")

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

    def on_chunk_callback(chunk_artifact: "WorkPieceArtifact"):
        """
        Callback to handle intermediate chunks. Serializes the chunk to
        shared memory and sends an event to the main process.
        """
        if chunk_artifact.ops.is_empty():
            return

        chunk_handle = artifact_store.put(
            chunk_artifact, creator_tag=f"{creator_tag}_chunk"
        )
        acked = proxy.send_event_and_wait(
            "visual_chunk_ready",
            {
                "handle_dict": chunk_handle.to_dict(),
                "generation_id": generation_id,
            },
            logger=logger,
        )
        if acked:
            artifact_store.forget(chunk_handle)
        else:
            logger.warning(
                "Chunk not acknowledged (NACK/Timeout). "
                "Releasing handle to prevent leak."
            )
            artifact_store.release(chunk_handle)

    final_artifact = compute_workpiece_artifact(
        workpiece=workpiece,
        opsproducer=opsproducer,
        laser=laser,
        transformers=opstransformers,
        settings=settings,
        pixels_per_mm=pixels_per_mm,
        generation_size=generation_size,
        generation_id=generation_id,
        on_chunk=on_chunk_callback,
        context=context,
    )

    if final_artifact is None:
        # If no artifact was produced (e.g., empty image), we still need
        # to return the generation_id to signal completion.
        proxy.send_event(
            "artifact_created",
            {"handle_dict": None, "generation_id": generation_id},
        )
        return generation_id

    # 1. Put artifact into Shared Memory
    handle = artifact_store.put(final_artifact, creator_tag=creator_tag)

    # 2. Inter-process handoff: Send handle to main process and wait.
    # The main process will adopt the handle via
    # ArtifactManager.safe_adoption().
    # On Windows, shared memory is destroyed when all handles are closed,
    # so we must wait for adoption before forgetting our reference.
    acked = proxy.send_event_and_wait(
        "artifact_created",
        {"handle_dict": handle.to_dict(), "generation_id": generation_id},
        logger=logger,
    )

    # 3. Forget (close without unlinking) after main process has adopted.
    # This transfers ownership: main process now owns the lifecycle.
    if acked:
        logger.debug("Artifact acknowledged, forgetting handle.")
        artifact_store.forget(handle)
    else:
        logger.warning(
            "Artifact not acknowledged (NACK/Timeout). Releasing handle "
            "to prevent leak."
        )
        artifact_store.release(handle)

    return generation_id
