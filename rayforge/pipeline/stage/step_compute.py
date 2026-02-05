from __future__ import annotations
import logging
import math
from typing import List, Tuple, Optional, TYPE_CHECKING
from ...core.ops import Ops
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from ...shared.tasker.progress import ProgressContext
from ..artifact import (
    StepRenderArtifact,
    StepOpsArtifact,
    WorkPieceArtifact,
)
from ..artifact.base import TextureData, TextureInstance
from ..encoder.textureencoder import TextureEncoder
from ..encoder.vertexencoder import VertexEncoder

if TYPE_CHECKING:
    from ..transformer import OpsTransformer

logger = logging.getLogger(__name__)


def compute_step_artifacts(
    artifacts: List[Tuple[WorkPieceArtifact, Matrix, WorkPiece]],
    transformers: List["OpsTransformer"],
    context: Optional[ProgressContext] = None,
) -> Tuple[StepRenderArtifact, StepOpsArtifact]:
    """
    Computes step artifacts from workpiece artifacts and transforms.

    Args:
        artifacts: List of tuples containing (WorkPieceArtifact, world_matrix,
                   workpiece) for each workpiece in the assembly.
        transformers: List of OpsTransformers to apply to the combined ops.
        context: Optional ProgressContext for progress reporting.

    Returns:
        Tuple of (StepRenderArtifact, StepOpsArtifact).
    """
    combined_ops = Ops()
    texture_instances = []
    num_items = len(artifacts)

    for i, (artifact, world_matrix, workpiece) in enumerate(artifacts):
        if context:
            context.set_progress(i / num_items * 0.5)

        ops = artifact.ops.copy()

        (tx, ty, angle, sx, sy, skew) = world_matrix.decompose()

        if artifact.is_scalable:
            if artifact.source_dimensions:
                target_w, target_h = workpiece.size
                source_w, source_h = artifact.source_dimensions
                if source_w > 1e-9 and source_h > 1e-9:
                    ops.scale(target_w / source_w, target_h / source_h)

        workpiece_placement_matrix = Matrix.compose(
            tx, ty, angle, 1.0, math.copysign(1.0, sy), skew
        )
        ops.transform(workpiece_placement_matrix.to_4x4_numpy())

        combined_ops.extend(ops)

        chunk_texture_data = None
        if not artifact.is_scalable:
            encoder_texture = TextureEncoder()
            if artifact.source_dimensions:
                width_px = int(artifact.source_dimensions[0])
                height_px = int(artifact.source_dimensions[1])
                px_per_mm_x = width_px / artifact.generation_size[0]
                px_per_mm_y = height_px / artifact.generation_size[1]
            else:
                px_per_mm_x = px_per_mm_y = 50.0
                width_px = int(
                    round(artifact.generation_size[0] * px_per_mm_x)
                )
                height_px = int(
                    round(artifact.generation_size[1] * px_per_mm_y)
                )

            if width_px > 0 and height_px > 0:
                texture_buffer = encoder_texture.encode(
                    artifact.ops,
                    width_px,
                    height_px,
                    (px_per_mm_x, px_per_mm_y),
                )
                chunk_texture_data = TextureData(
                    power_texture_data=texture_buffer,
                    dimensions_mm=artifact.generation_size,
                    position_mm=(0.0, 0.0),
                )

        if chunk_texture_data is not None:
            chunk_w_mm, chunk_h_mm = chunk_texture_data.dimensions_mm
            chunk_x_off, chunk_y_off = chunk_texture_data.position_mm

            chunk_scale_matrix = Matrix.scale(chunk_w_mm, chunk_h_mm)

            local_translation_matrix = Matrix.translation(
                chunk_x_off, chunk_y_off
            )

            final_transform = (
                workpiece_placement_matrix
                @ local_translation_matrix
                @ chunk_scale_matrix
            )

            instance = TextureInstance(
                texture_data=chunk_texture_data,
                world_transform=final_transform.to_4x4_numpy(),
            )
            texture_instances.append(instance)

    for i, transformer in enumerate(transformers):
        if context:
            context.set_message(
                _("Applying '{t}'").format(t=transformer.label)
            )
            base_progress = 0.5 + (i / len(transformers) * 0.4)
            context.set_progress(base_progress)
        transformer.run(combined_ops)

    if context:
        context.set_progress(0.9)

    encoder = VertexEncoder()
    vertex_data = encoder.encode(combined_ops)

    if context:
        context.set_progress(0.95)

    render_artifact = StepRenderArtifact(
        vertex_data=vertex_data, texture_instances=texture_instances
    )
    ops_artifact = StepOpsArtifact(ops=combined_ops)

    if context:
        context.set_progress(1.0)

    return render_artifact, ops_artifact
