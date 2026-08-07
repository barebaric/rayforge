from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ...pipeline.artifact.handle import create_handle_from_dict
from ...pipeline.artifact.job import JobArtifact
from ...pipeline.artifact.store import ArtifactStore
from .compiled_scene import CompiledSceneArtifact
from .render_config import RenderConfig3D
from .scene_compiler import compile_scene

logger = logging.getLogger(__name__)


def compile_scene_from_job(
    artifact_store: ArtifactStore,
    job_handle_dict: Dict[str, Any],
    render_config_dict: dict,
) -> Optional[CompiledSceneArtifact]:
    """Compile a 3D scene from a job artifact.

    Runs synchronously on the calling thread; the caller owns threading.
    The compiled artifact is returned directly, avoiding pickling of
    raygeo ``Ops`` objects through multiprocessing queues.
    """
    config = RenderConfig3D.from_dict(render_config_dict)

    try:
        handle = create_handle_from_dict(job_handle_dict)
        artifact = artifact_store.get(handle)
    except Exception:
        logger.warning("Job artifact no longer available. Aborting.")
        return None

    if not isinstance(artifact, JobArtifact):
        logger.error(f"Expected JobArtifact, got {type(artifact).__name__}.")
        return None

    ops = artifact.preview_ops
    if ops is None or ops.is_empty():
        logger.debug("Job artifact ops are empty.")
        return None

    t_start = time.perf_counter()
    compiled = compile_scene(ops, config)
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.debug(f"Compilation took {elapsed:.1f}ms (commands={len(ops)})")
    return compiled
