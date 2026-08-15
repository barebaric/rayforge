from __future__ import annotations

import logging
import time
from typing import Any

from ...pipeline.artifact.handle import create_handle_from_dict
from ...pipeline.artifact.job import JobArtifact
from ...pipeline.artifact.store import ArtifactStore
from .compiled_scene import CompiledSceneArtifact
from .render_config import RenderConfig3D
from .scene_compiler import compile_scene
from .stock_compiler import compile_stock_layers

logger = logging.getLogger(__name__)


def compile_scene_from_job(
    artifact_store: ArtifactStore,
    job_handle_dict: dict[str, Any],
    render_config_dict: dict,
) -> CompiledSceneArtifact | None:
    """Compile a 3D scene from a job artifact.

    Runs synchronously on the calling thread; the caller owns threading.
    The compiled artifact is returned directly, avoiding pickling of
    raygeo ``Ops`` objects through multiprocessing queues.
    """
    config = RenderConfig3D.from_dict(render_config_dict)

    try:
        handle = create_handle_from_dict(job_handle_dict)
        artifact = artifact_store.get(handle)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.warning(f"Job artifact no longer available. Aborting: {e}")
        return None

    if not isinstance(artifact, JobArtifact):
        logger.error(f"Expected JobArtifact, got {type(artifact).__name__}.")
        return None

    ops = artifact.preview_ops
    if ops is None or ops.is_empty():
        logger.debug("Job artifact ops are empty.")
        return None

    t_start = time.perf_counter()
    compiled = compile_scene(ops, config, generation_id=artifact.generation_id)
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.debug(f"Compilation took {elapsed:.1f}ms (commands={len(ops)})")
    return compiled


def compile_stock_scene(
    render_config_dict: dict,
) -> CompiledSceneArtifact | None:
    """Compile a stock-only 3D scene without a job artifact.

    Used when the document has no assembled job (e.g. no visible
    steps with workpieces) so the configured stock still renders.
    Runs synchronously on the calling thread; the caller owns
    threading.
    """
    config = RenderConfig3D.from_dict(render_config_dict)
    stock_layers = compile_stock_layers(
        config.stock_specs or [], config.world_to_visual
    )
    if not stock_layers:
        return None
    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=[],
        texture_layers=[],
        overlay_layers=[],
        laser_uid_order=[],
        stock_layers=stock_layers,
    )
