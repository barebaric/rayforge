from __future__ import annotations
import logging
import math
from typing import List, Dict, Any, Optional
from .artifact import (
    ArtifactStore,
    StepArtifact,
    create_handle_from_dict,
    WorkPieceArtifact,
    TextureData,
)
from .coord import CoordinateSystem
from ..core.ops import Ops
from ..core.matrix import Matrix
from ..core.workpiece import WorkPiece
from ..shared.tasker.proxy import ExecutionContextProxy
from ..pipeline.transformer import OpsTransformer, transformer_by_name
from ..pipeline.encoder.vertexencoder import VertexEncoder

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


def run_step_assembly_in_subprocess(
    proxy: ExecutionContextProxy,
    workpiece_assembly_info: List[Dict[str, Any]],
    step_uid: str,
    generation_id: int,
    per_step_transformers_dicts: List[Dict[str, Any]],
) -> Optional[tuple[Dict[str, Any], int]]:
    """
    Aggregates WorkPieceArtifacts into a final StepArtifact in a background
    process.
    """
    proxy.set_message(_("Assembling step..."))
    logger.debug(f"Starting step assembly for step_uid: {step_uid}")

    if not workpiece_assembly_info:
        logger.warning("No workpiece info provided for step assembly.")
        return None

    combined_ops = Ops()
    aggregated_texture: Optional[TextureData] = None
    num_items = len(workpiece_assembly_info)

    for i, info in enumerate(workpiece_assembly_info):
        proxy.set_progress(i / num_items * 0.5)

        handle = create_handle_from_dict(info["artifact_handle_dict"])
        artifact = ArtifactStore.get(handle)
        if not isinstance(artifact, WorkPieceArtifact):
            continue

        workpiece = WorkPiece.from_dict(info["workpiece_dict"])
        ops = artifact.ops.copy()

        # 1. Ensure ops are in final local size (in mm).
        if artifact.is_scalable and artifact.source_dimensions:
            target_w, target_h = workpiece.size
            source_w, source_h = artifact.source_dimensions
            if source_w > 1e-9 and source_h > 1e-9:
                ops.scale(target_w / source_w, target_h / source_h)

        # 2. Apply a PLACEMENT-ONLY transform to move to world space.
        # This logic is adapted from the original, correct canvas3d logic.
        world_matrix = Matrix.from_list(info["world_transform_list"])
        (tx, ty, angle, sx, sy, skew) = world_matrix.decompose()
        placement_matrix = Matrix.compose(
            tx, ty, angle, 1.0, math.copysign(1.0, sy), skew
        )
        ops.transform(placement_matrix.to_4x4_numpy())
        combined_ops.extend(ops)

        # 3. Handle and transform texture data.
        # For now, we handle one texture per step.
        if artifact.texture_data and aggregated_texture is None:
            wx, wy, ww, wh = workpiece.bbox
            aggregated_texture = TextureData(
                power_texture_data=artifact.texture_data.power_texture_data,
                dimensions_mm=(ww, wh),
                position_mm=(wx, wy),
            )

    # 4. Apply per-step transformers to the world-space ops
    transformers = _instantiate_transformers(per_step_transformers_dicts)
    for i, transformer in enumerate(transformers):
        base_progress = 0.5 + (i / len(transformers) * 0.4)
        progress_range = 0.4 / len(transformers)
        sub_proxy = proxy.sub_context(base_progress, progress_range)
        proxy.set_message(_("Applying '{t}'").format(t=transformer.label))
        transformer.run(combined_ops, context=sub_proxy)

    proxy.set_progress(0.9)
    # 5. Generate final vertex data for 3D rendering
    proxy.set_message(_("Encoding for 3D preview..."))
    encoder = VertexEncoder()
    vertex_data = encoder.encode(combined_ops)
    proxy.set_progress(0.95)

    # 6. Create and store the final StepArtifact
    proxy.set_message(_("Storing final step data..."))
    final_artifact = StepArtifact(
        ops=combined_ops,
        is_scalable=False,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        vertex_data=vertex_data,
        texture_data=aggregated_texture,
    )
    final_handle = ArtifactStore.put(final_artifact)
    proxy.set_progress(1.0)
    logger.debug(f"Step assembly for {step_uid} complete.")

    return final_handle.to_dict(), generation_id
