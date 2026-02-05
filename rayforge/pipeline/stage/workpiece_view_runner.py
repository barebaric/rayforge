from __future__ import annotations
import logging
import numpy as np
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Optional, Dict, Any
from ...shared.tasker.progress import CallbackProgressContext
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact import (
    WorkPieceArtifact,
    create_handle_from_dict,
)
from ..artifact.manager import ArtifactManager
from ..artifact.store import ArtifactStore
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
)
from .workpiece_view_compute import (
    compute_workpiece_view_to_buffer,
    compute_view_dimensions,
    render_chunk_to_buffer,
)

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)


def render_chunk_to_view(
    artifact_manager: ArtifactManager,
    chunk_handle_dict: Dict[str, Any],
    view_handle_dict: Dict[str, Any],
    render_context_dict: Dict[str, Any],
) -> bool:
    """
    Renders a chunk artifact to a view artifact in the current process.

    This is the Runner layer function that handles chunk rendering.
    It loads the chunk and view artifacts, opens the shared memory,
    and calls the compute function to perform rendering.

    Args:
        artifact_manager: The artifact manager for loading artifacts.
        chunk_handle_dict: Dictionary representation of the chunk
            WorkPieceArtifactHandle to load.
        view_handle_dict: Dictionary representation of the view
            WorkPieceViewArtifactHandle to load.
        render_context_dict: Dictionary representation of the
            RenderContext.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    from ..artifact import create_handle_from_dict

    chunk_handle = create_handle_from_dict(chunk_handle_dict)
    view_handle = create_handle_from_dict(view_handle_dict)

    chunk_artifact = artifact_manager.get_artifact(chunk_handle)
    if not isinstance(chunk_artifact, WorkPieceArtifact):
        logger.error("Runner received incorrect chunk artifact type.")
        return False

    view_artifact = artifact_manager.get_artifact(view_handle)
    if not isinstance(view_artifact, WorkPieceViewArtifact):
        logger.error("Runner could not retrieve view artifact.")
        return False

    context = RenderContext.from_dict(render_context_dict)
    view_bbox_mm = view_artifact.bbox_mm

    shm = None
    try:
        shm = shared_memory.SharedMemory(name=view_handle.shm_name)
        height_px, width_px = view_artifact.bitmap_data.shape[:2]
        shm_bitmap = np.ndarray(
            shape=(height_px, width_px, 4),
            dtype=np.uint8,
            buffer=shm.buf,
        )

        return render_chunk_to_buffer(
            chunk_artifact, context, shm_bitmap, view_bbox_mm
        )
    finally:
        if shm:
            shm.close()


def make_workpiece_view_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    workpiece_artifact_handle_dict: Dict[str, Any],
    render_context_dict: Dict[str, Any],
    creator_tag: str,
    adoption_event: Optional["threading.Event"] = None,
) -> Optional[Dict[str, Any]]:
    """
    Renders a WorkPieceArtifact to a bitmap in a background process.

    This is the boundary layer that handles:
    - Loading the WorkPieceArtifact from artifact store
    - Hydrating the RenderContext from its dictionary representation
    - Creating the view artifact and handle upfront
    - Calling the compute function to perform rendering into shared memory
    - Sending progress events via the proxy

    Args:
        proxy: The execution context proxy for sending events.
        artifact_store: The artifact store for loading and storing artifacts.
        workpiece_artifact_handle_dict: Dictionary representation of the
            WorkPieceArtifactHandle to load.
        render_context_dict: Dictionary representation of the RenderContext.
        creator_tag: Tag for artifact creation tracking.
        adoption_event: Optional event for artifact adoption signaling.

    Returns:
        None (result is communicated via events).
    """
    logger.debug("Worker: Starting view artifact rendering...")

    handle = create_handle_from_dict(workpiece_artifact_handle_dict)
    logger.debug(f"Worker: Retrieved handle {handle.shm_name}")
    artifact = artifact_store.get(handle)
    if not isinstance(artifact, WorkPieceArtifact):
        logger.error("Runner received incorrect artifact type.")
        return None

    context = RenderContext.from_dict(render_context_dict)

    logger.debug("Worker: Calculating dimensions...")
    dims = compute_view_dimensions(artifact, context)
    if dims is None:
        logger.warning("Worker: No content to render. Returning None.")
        return None

    x_mm, y_mm, w_mm, h_mm, width_px, height_px = dims
    bbox = (x_mm, y_mm, w_mm, h_mm)

    logger.debug(
        f"Worker: Creating view artifact with dimensions "
        f"{width_px}x{height_px}"
    )
    bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)

    from ..artifact.workpiece_view import WorkPieceViewArtifact

    view_artifact = WorkPieceViewArtifact(bitmap_data=bitmap, bbox_mm=bbox)
    view_handle = artifact_store.put(view_artifact, creator_tag=creator_tag)
    logger.debug(f"Worker: Created view artifact {view_handle.shm_name}")

    proxy.send_event(
        "view_artifact_created",
        {"handle_dict": view_handle.to_dict()},
    )
    logger.debug("Worker: Sent view_artifact_created event")

    shm = None
    try:
        logger.debug("Worker: Opening shared memory for rendering...")
        shm = shared_memory.SharedMemory(name=view_handle.shm_name)
        shm_bitmap = np.ndarray(
            shape=(height_px, width_px, 4),
            dtype=np.uint8,
            buffer=shm.buf,
        )

        def _send_update_event():
            proxy.send_event("view_artifact_updated")
            logger.debug("Worker: Sent view_artifact_updated event")

        progress_context = CallbackProgressContext(
            is_cancelled_func=proxy.is_cancelled,
            progress_callback=lambda p: (
                proxy.set_progress(p),
                _send_update_event(),
            )[-1],
            message_callback=proxy.set_message,
        )

        result_bbox = compute_workpiece_view_to_buffer(
            artifact, context, shm_bitmap, progress_context
        )

        if result_bbox is None:
            logger.warning(
                "Worker: compute_workpiece_view_to_buffer returned None."
            )
            return None

    finally:
        if shm:
            shm.close()

    proxy.send_event("view_artifact_updated")
    logger.debug("Worker: Sent final view_artifact_updated event")

    logger.debug("Worker: View artifact rendering complete.")
    proxy.set_progress(1.0)

    return None
