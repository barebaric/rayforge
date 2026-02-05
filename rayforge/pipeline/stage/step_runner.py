from __future__ import annotations
import logging
import threading
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from ...shared.tasker.progress import CallbackProgressContext
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact import (
    create_handle_from_dict,
    WorkPieceArtifact,
)
from ..artifact.store import ArtifactStore
from ..transformer import OpsTransformer, transformer_by_name

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _instantiate_transformers(
    transformer_dicts: List[Dict[str, Any]],
) -> List[OpsTransformer]:
    """Helper to create transformer instances from a list of dicts."""
    transformers: List[OpsTransformer] = []
    for t_dict in transformer_dicts:
        if not t_dict.get("enabled", True):
            continue
        cls_name = t_dict.get("name")
        if cls_name and cls_name in transformer_by_name:
            cls = transformer_by_name[cls_name]
            try:
                transformers.append(cls.from_dict(t_dict))
            except Exception as e:
                logger.error(
                    f"Failed to instantiate transformer '{cls_name}': {e}",
                    exc_info=True,
                )
    return transformers


def make_step_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    workpiece_assembly_info: List[Dict[str, Any]],
    step_uid: str,
    generation_id: int,
    per_step_transformers_dicts: List[Dict[str, Any]],
    cut_speed: float,
    travel_speed: float,
    acceleration: float,
    creator_tag: str,
    adoption_event: Optional[threading.Event] = None,
) -> int:
    """
    Aggregates WorkPieceArtifacts, creates a StepArtifact, sends its handle
    back via an event, and then returns the final generation ID.
    """
    from .step_compute import compute_step_artifacts

    proxy.set_message(_("Assembling step..."))
    logger.debug(f"Starting step assembly for step_uid: {step_uid}")

    if not workpiece_assembly_info:
        logger.warning("No workpiece info provided for step assembly.")
        return generation_id

    artifacts = []
    num_items = len(workpiece_assembly_info)

    for i, info in enumerate(workpiece_assembly_info):
        proxy.set_progress(i / num_items * 0.5)

        handle = create_handle_from_dict(info["artifact_handle_dict"])
        artifact = artifact_store.get(handle)
        if not isinstance(artifact, WorkPieceArtifact):
            continue

        workpiece = WorkPiece.from_dict(info["workpiece_dict"])
        world_matrix = Matrix.from_list(info["world_transform_list"])

        artifacts.append((artifact, world_matrix, workpiece))

    transformers = _instantiate_transformers(per_step_transformers_dicts)

    context = CallbackProgressContext(
        is_cancelled_func=proxy.is_cancelled,
        progress_callback=proxy.set_progress,
        message_callback=proxy.set_message,
    )

    render_artifact, ops_artifact = compute_step_artifacts(
        artifacts=artifacts,
        transformers=transformers,
        context=context,
    )

    proxy.set_message(_("Storing step data..."))

    render_handle = artifact_store.put(
        render_artifact, creator_tag=f"{creator_tag}_render"
    )

    # Send render handle back via event for instant UI update
    proxy.send_event(
        "render_artifact_ready",
        {
            "handle_dict": render_handle.to_dict(),
            "generation_id": generation_id,
        },
    )

    ops_handle = artifact_store.put(
        ops_artifact, creator_tag=f"{creator_tag}_ops"
    )
    proxy.send_event(
        "ops_artifact_ready",
        {
            "handle_dict": ops_handle.to_dict(),
            "generation_id": generation_id,
        },
    )

    # Wait for main process to adopt both artifacts before forgetting them
    if adoption_event is not None:
        logger.debug(
            "Waiting for main process to adopt step artifacts for "
            f"{step_uid}..."
        )
        if adoption_event.wait(timeout=10):
            logger.debug(
                "Main process adopted step artifacts for "
                f"{step_uid}. Forgetting..."
            )
            artifact_store.forget(render_handle)
            artifact_store.forget(ops_handle)
            logger.info(
                f"Worker disowned step artifacts for {step_uid} successfully"
            )
        else:
            logger.warning(
                "Main process failed to adopt step artifacts for "
                f"{step_uid} within timeout. Releasing to prevent leak."
            )
            artifact_store.release(render_handle)
            artifact_store.release(ops_handle)

    # 7. Calculate time estimate
    proxy.set_message(_("Calculating time estimate..."))
    final_time = ops_artifact.ops.estimate_time(
        default_cut_speed=cut_speed,
        default_travel_speed=travel_speed,
        acceleration=acceleration,
    )
    proxy.send_event(
        "time_estimate_ready",
        {"time_estimate": final_time, "generation_id": generation_id},
    )

    proxy.set_progress(1.0)
    logger.debug(f"Step assembly for {step_uid} complete.")

    # 8. Return the generation ID to signal completion
    return generation_id
