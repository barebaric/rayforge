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
from ..artifact.base import TextureData, TextureInstance, VertexData
from ..encoder.textureencoder import TextureEncoder
from ..encoder.vertexencoder import VertexEncoder

if TYPE_CHECKING:
    from ..transformer import OpsTransformer

logger = logging.getLogger(__name__)


def _apply_artifact_scaling(
    ops: Ops, artifact: WorkPieceArtifact, workpiece: WorkPiece
) -> None:
    """
    Applies scaling to ops for scalable artifacts.

    Args:
        ops: The ops to scale.
        artifact: The workpiece artifact.
        workpiece: The workpiece containing target dimensions.
    """
    if artifact.is_scalable and artifact.source_dimensions:
        target_w, target_h = workpiece.size
        source_w, source_h = artifact.source_dimensions
        if source_w > 1e-9 and source_h > 1e-9:
            ops.scale(target_w / source_w, target_h / source_h)


def _create_workpiece_placement_matrix(
    tx: float, ty: float, angle: float, sy: float, skew: float
) -> Matrix:
    """
    Creates the workpiece placement matrix.

    Args:
        tx: Translation X.
        ty: Translation Y.
        angle: Rotation angle.
        sy: Scale Y (sign determines flip).
        skew: Skew angle.

    Returns:
        The workpiece placement matrix.
    """
    return Matrix.compose(tx, ty, angle, 1.0, math.copysign(1.0, sy), skew)


def _calculate_texture_dimensions(
    artifact: WorkPieceArtifact,
) -> Tuple[int, int, float, float]:
    """
    Calculates texture dimensions and pixels per mm.

    Args:
        artifact: The workpiece artifact.

    Returns:
        Tuple of (width_px, height_px, px_per_mm_x, px_per_mm_y).
    """
    if artifact.source_dimensions:
        width_px = int(artifact.source_dimensions[0])
        height_px = int(artifact.source_dimensions[1])
        px_per_mm_x = width_px / artifact.generation_size[0]
        px_per_mm_y = height_px / artifact.generation_size[1]
    else:
        px_per_mm_x = px_per_mm_y = 50.0
        width_px = int(round(artifact.generation_size[0] * px_per_mm_x))
        height_px = int(round(artifact.generation_size[1] * px_per_mm_y))
    return width_px, height_px, px_per_mm_x, px_per_mm_y


def _create_texture_data(
    artifact: WorkPieceArtifact,
) -> Optional[TextureData]:
    """
    Creates texture data for non-scalable artifacts.

    Args:
        artifact: The workpiece artifact.

    Returns:
        TextureData if created, None otherwise.
    """
    if artifact.is_scalable:
        return None

    width_px, height_px, px_per_mm_x, px_per_mm_y = (
        _calculate_texture_dimensions(artifact)
    )

    if width_px > 0 and height_px > 0:
        encoder_texture = TextureEncoder()
        texture_buffer = encoder_texture.encode(
            artifact.ops,
            width_px,
            height_px,
            (px_per_mm_x, px_per_mm_y),
        )
        return TextureData(
            power_texture_data=texture_buffer,
            dimensions_mm=artifact.generation_size,
            position_mm=(0.0, 0.0),
        )
    return None


def _create_texture_instance(
    chunk_texture_data: TextureData,
    workpiece_placement_matrix: Matrix,
) -> TextureInstance:
    """
    Creates a texture instance from texture data.

    Args:
        chunk_texture_data: The texture data.
        workpiece_placement_matrix: The workpiece placement matrix.

    Returns:
        The texture instance.
    """
    chunk_w_mm, chunk_h_mm = chunk_texture_data.dimensions_mm
    chunk_x_off, chunk_y_off = chunk_texture_data.position_mm

    chunk_scale_matrix = Matrix.scale(chunk_w_mm, chunk_h_mm)
    local_translation_matrix = Matrix.translation(chunk_x_off, chunk_y_off)

    final_transform = (
        workpiece_placement_matrix
        @ local_translation_matrix
        @ chunk_scale_matrix
    )

    return TextureInstance(
        texture_data=chunk_texture_data,
        world_transform=final_transform.to_4x4_numpy(),
    )


def _process_artifact(
    artifact: WorkPieceArtifact,
    world_matrix: Matrix,
    workpiece: WorkPiece,
) -> Tuple[Ops, Optional[TextureInstance]]:
    """
    Processes a single artifact and returns ops and optional texture instance.

    Args:
        artifact: The workpiece artifact.
        world_matrix: The world transformation matrix.
        workpiece: The workpiece.

    Returns:
        Tuple of (ops, texture_instance).
    """
    ops = artifact.ops.copy()
    _apply_artifact_scaling(ops, artifact, workpiece)

    tx, ty, angle, sx, sy, skew = world_matrix.decompose()
    workpiece_placement_matrix = _create_workpiece_placement_matrix(
        tx, ty, angle, sy, skew
    )
    ops.transform(workpiece_placement_matrix.to_4x4_numpy())

    texture_instance = None
    if not artifact.is_scalable:
        chunk_texture_data = _create_texture_data(artifact)
        if chunk_texture_data is not None:
            texture_instance = _create_texture_instance(
                chunk_texture_data, workpiece_placement_matrix
            )

    return ops, texture_instance


def _apply_transformers_to_ops(
    ops: Ops,
    transformers: List["OpsTransformer"],
    context: Optional[ProgressContext] = None,
) -> None:
    """
    Applies transformers to ops.

    Args:
        ops: The ops to transform.
        transformers: List of transformers to apply.
        context: Optional progress context.
    """
    for i, transformer in enumerate(transformers):
        if context:
            context.set_message(
                _("Applying '{t}'").format(t=transformer.label)
            )
            base_progress = 0.5 + (i / len(transformers) * 0.4)
            context.set_progress(base_progress)
        transformer.run(ops)


def _encode_vertex_data(ops: Ops) -> VertexData:
    """
    Encodes ops to vertex data.

    Args:
        ops: The ops to encode.

    Returns:
        The encoded vertex data.
    """
    encoder = VertexEncoder()
    return encoder.encode(ops)


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

        ops, texture_instance = _process_artifact(
            artifact, world_matrix, workpiece
        )
        combined_ops.extend(ops)
        if texture_instance is not None:
            texture_instances.append(texture_instance)

    _apply_transformers_to_ops(combined_ops, transformers, context)

    if context:
        context.set_progress(0.9)

    vertex_data = _encode_vertex_data(combined_ops)

    if context:
        context.set_progress(0.95)

    render_artifact = StepRenderArtifact(
        vertex_data=vertex_data, texture_instances=texture_instances
    )
    ops_artifact = StepOpsArtifact(ops=combined_ops)

    if context:
        context.set_progress(1.0)

    return render_artifact, ops_artifact
