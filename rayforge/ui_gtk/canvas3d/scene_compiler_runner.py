from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional

from ...pipeline.artifact.handle import create_handle_from_dict
from ...pipeline.artifact.store import ArtifactStore, SharedMemoryNotFoundError
from ...pipeline.artifact.step_ops import StepOpsArtifact
from ...shared.tasker.proxy import ExecutionContextProxy
from .render_config import RenderConfig3D
from .scene_compiler import compile_scene

logger = logging.getLogger(__name__)


def compile_scene_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    step_ops_handle_dicts: List,
    render_config_dict: dict,
) -> Optional[Dict[str, Any]]:
    config = RenderConfig3D.from_dict(render_config_dict)

    ops_list: list = []
    for step_uid, handle_dict in step_ops_handle_dicts:
        if proxy.is_cancelled():
            return None

        step_cfg = config.step_configs.get(step_uid)
        if step_cfg is None:
            logger.debug(f"No config for step {step_uid}, skipping.")
            continue

        try:
            handle = create_handle_from_dict(handle_dict)
            artifact = artifact_store.get(handle)
        except (SharedMemoryNotFoundError, RuntimeError) as e:
            logger.warning(
                f"Step ops SHM no longer available for {step_uid}: {e}. "
                "Skipping."
            )
            continue
        except Exception as e:
            logger.error(f"Failed to load ops for step {step_uid}: {e}")
            continue

        if not isinstance(artifact, StepOpsArtifact):
            logger.error(f"Artifact for {step_uid} is not StepOpsArtifact.")
            continue

        if artifact.ops.is_empty():
            continue

        ops_list.append((artifact.ops, step_cfg))

    if not ops_list:
        logger.debug("No ops to compile.")
        return None

    t_start = time.perf_counter()
    compiled = compile_scene(ops_list, config)
    elapsed = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[SCENE_COMPILER] Compilation took {elapsed:.1f}ms "
        f"(steps={len(ops_list)})"
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
