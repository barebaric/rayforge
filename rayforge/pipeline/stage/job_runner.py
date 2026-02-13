from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

from ...core.doc import Doc
from ...machine.models.machine import Machine
from ...shared.tasker.progress import CallbackProgressContext
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact import (
    create_handle_from_dict,
    StepOpsArtifact,
)
from ..artifact.key import ArtifactKey
from ..artifact.store import ArtifactStore
from .job_compute import compute_job_artifact

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@dataclass
class JobDescription:
    """A complete, serializable description of a job for the subprocess."""

    step_artifact_handles_by_uid: Dict[str, Dict[str, Any]]
    machine_dict: Dict[str, Any]
    doc_dict: Dict[str, Any]


def make_job_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    job_description_dict: Dict[str, Any],
    creator_tag: str,
    generation_id: int,
    job_key: ArtifactKey,
) -> None:
    """
    The main entry point for assembling, post-processing, and encoding a
    full job in a background process.

    Uses acknowledgment-based handover for shared memory artifacts.
    """
    job_desc = JobDescription(**job_description_dict)
    machine = Machine.from_dict(job_desc.machine_dict, is_inert=True)
    doc = Doc.from_dict(job_desc.doc_dict)
    handles_by_uid = job_desc.step_artifact_handles_by_uid

    step_artifacts_by_uid: Dict[str, StepOpsArtifact] = {}

    for step_uid, handle_dict in handles_by_uid.items():
        handle = create_handle_from_dict(handle_dict)
        artifact = artifact_store.get(handle)
        if isinstance(artifact, StepOpsArtifact):
            step_artifacts_by_uid[step_uid] = artifact

    context = CallbackProgressContext(
        is_cancelled_func=proxy.is_cancelled,
        progress_callback=proxy.set_progress,
        message_callback=proxy.set_message,
    )

    final_artifact = compute_job_artifact(
        doc,
        step_artifacts_by_uid,
        machine,
        context,
    )

    proxy.set_message(_("Storing final job artifact..."))
    final_handle = artifact_store.put(final_artifact, creator_tag=creator_tag)

    acked = proxy.send_event_and_wait(
        "artifact_created",
        {
            "handle_dict": final_handle.to_dict(),
            "generation_id": generation_id,
            "job_key": {"id": job_key.id, "group": job_key.group},
        },
        logger=logger,
    )

    if acked:
        logger.debug("Job artifact acknowledged, forgetting handle.")
        artifact_store.forget(final_handle)
    else:
        logger.warning("Job artifact not acknowledged, keeping handle open")

    proxy.set_progress(1.0)
    proxy.set_message(_("Job finalization complete"))
    return
