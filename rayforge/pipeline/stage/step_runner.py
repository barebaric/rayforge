from __future__ import annotations
import logging
import math
from typing import List, Dict, Any
from ...context import get_context
from ...core.ops import Ops
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from ...shared.tasker.proxy import ExecutionContextProxy
from ..artifact import (
    StepRenderArtifact,
    StepOpsArtifact,
    create_handle_from_dict,
    WorkPieceArtifact,
)
from ..artifact.base import TextureInstance
from ..encoder.vertexencoder import VertexEncoder
from ..transformer import OpsTransformer, transformer_by_name

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
    workpiece_assembly_info: List[Dict[str, Any]],
    step_uid: str,
    generation_id: int,
    per_step_transformers_dicts: List[Dict[str, Any]],
    cut_speed: float,
    travel_speed: float,
    acceleration: float,
    creator_tag: str,
) -> int:
    """
    Aggregates WorkPieceArtifacts, creates a StepArtifact, sends its handle
    back via an event, and then returns the final generation ID.
    """
    proxy.set_message(_("Assembling step..."))
    logger.debug(f"Starting step assembly for step_uid: {step_uid}")

    if not workpiece_assembly_info:
        logger.warning("No workpiece info provided for step assembly.")
        return generation_id

    artifact_store = get_context().artifact_store
    combined_ops = Ops()
    texture_instances = []
    num_items = len(workpiece_assembly_info)

    for i, info in enumerate(workpiece_assembly_info):
        proxy.set_progress(i / num_items * 0.5)

        handle = create_handle_from_dict(info["artifact_handle_dict"])
        artifact = artifact_store.get(handle)
        if not isinstance(artifact, WorkPieceArtifact):
            continue

        workpiece = WorkPiece.from_dict(info["workpiece_dict"])
        ops = artifact.ops.copy()

        # Decompose the world matrix to separate Scale from Placement
        world_matrix = Matrix.from_list(info["world_transform_list"])
        (tx, ty, angle, sx, sy, skew) = world_matrix.decompose()

        # 1. Scaling Phase: Only for scalable artifacts (e.g. Vectors/Sketches)
        #    Non-scalable artifacts (Raster/Contour) are already sized in mm.
        if artifact.is_scalable:
            if artifact.source_dimensions:
                target_w, target_h = workpiece.size
                source_w, source_h = artifact.source_dimensions
                if source_w > 1e-9 and source_h > 1e-9:
                    ops.scale(target_w / source_w, target_h / source_h)

        # 2. Placement Phase: Apply Translation and Rotation to ALL artifacts.
        #    This moves them from Local MM Space -> World Space.
        #    Note: We reconstruct the matrix with Scale=1.0 because scaling
        #    was either handled above or pre-baked into the artifact.
        workpiece_placement_matrix = Matrix.compose(
            tx, ty, angle, 1.0, math.copysign(1.0, sy), skew
        )
        ops.transform(workpiece_placement_matrix.to_4x4_numpy())

        combined_ops.extend(ops)

        # 3. FOR TEXTURES: The renderer draws a 1x1 unit quad. We must
        # build a transform to scale it to the chunk's physical size,
        # place it locally, and then place it in the world.
        if artifact.texture_data:
            chunk_w_mm, chunk_h_mm = artifact.texture_data.dimensions_mm
            chunk_x_off, chunk_y_off = artifact.texture_data.position_mm

            # a) Create a matrix to scale the 1x1 unit quad to the chunk's
            # physical size in millimeters.
            chunk_scale_matrix = Matrix.scale(chunk_w_mm, chunk_h_mm)

            # b) Create a matrix to translate the correctly-sized chunk
            # to its position within the workpiece's local frame.
            local_translation_matrix = Matrix.translation(
                chunk_x_off, chunk_y_off
            )

            # c) The final transform combines these steps in order:
            # 1. Scale the unit quad to the chunk's physical size.
            # 2. Translate the chunk to its local position.
            # 3. Apply the workpiece's world PLACEMENT (no scale).
            final_transform = (
                workpiece_placement_matrix
                @ local_translation_matrix
                @ chunk_scale_matrix
            )

            instance = TextureInstance(
                texture_data=artifact.texture_data,
                world_transform=final_transform.to_4x4_numpy(),
            )
            texture_instances.append(instance)

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

    # 6. Create, store, and emit the new, separated artifacts
    proxy.set_message(_("Storing step data..."))

    # New, lightweight render artifact
    render_artifact = StepRenderArtifact(
        vertex_data=vertex_data, texture_instances=texture_instances
    )
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

    # Send ops handle back via a separate event
    ops_artifact = StepOpsArtifact(ops=combined_ops)
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

    # 7. Calculate time estimate
    proxy.set_message(_("Calculating time estimate..."))
    final_time = combined_ops.estimate_time(
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
