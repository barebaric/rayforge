from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional

from ...pipeline.artifact.handle import create_handle_from_dict
from ...pipeline.artifact.store import (
    ArtifactStore,
    SharedMemoryNotFoundError,
)
from ...pipeline.artifact.job import JobArtifact
from ...shared.tasker.proxy import ExecutionContextProxy
from .render_config import RenderConfig3D
from .scene_compiler import compile_scene

logger = logging.getLogger(__name__)


def compile_scene_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    job_handle_dict: Dict[str, Any],
    render_config_dict: dict,
) -> Optional[Dict[str, Any]]:
    config = RenderConfig3D.from_dict(render_config_dict)

    if proxy.is_cancelled():
        return None

    try:
        handle = create_handle_from_dict(job_handle_dict)
        artifact = artifact_store.get(handle)
    except (SharedMemoryNotFoundError, RuntimeError) as e:
        logger.warning(f"Job artifact SHM no longer available: {e}. Aborting.")
        return None
    except Exception as e:
        logger.error(f"Failed to load job artifact: {e}")
        return None

    if not isinstance(artifact, JobArtifact):
        logger.error(f"Expected JobArtifact, got {type(artifact).__name__}.")
        return None

    ops = artifact.mapped_ops if artifact.mapped_ops else artifact.ops
    if ops is None or ops.is_empty():
        logger.debug("Job artifact ops are empty.")
        return None

    t_start = time.perf_counter()
    compiled = compile_scene(ops, config)
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[SCENE_COMPILER] Compilation took {elapsed:.1f}ms "
        f"(commands={len(ops.commands)})"
    )

    compiled_handle = artifact_store.put(compiled, creator_tag="scene3d")
    logger.debug(f"Stored compiled scene: {compiled_handle.shm_name}")

    acked = proxy.send_event_and_wait(
        "scene_compiled",
        {"handle_dict": compiled_handle.to_dict()},
        logger=logger,
    )

    if acked:
        artifact_store.forget(compiled_handle)
    else:
        logger.warning("Scene artifact not acknowledged. Releasing handle.")
        artifact_store.release(compiled_handle)

    return None
