from __future__ import annotations
import logging
import math
from typing import List, Tuple, Optional, TYPE_CHECKING
from gettext import gettext as _
from ...core.ops import Ops
from ...core.matrix import Matrix
from ...core.workpiece import WorkPiece
from ...shared.tasker.progress import ProgressContext, set_progress
from ..artifact import (
    StepOpsArtifact,
    WorkPieceArtifact,
)

if TYPE_CHECKING:
    from ...core.geo import Geometry
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


def _process_artifact(
    artifact: WorkPieceArtifact,
    world_matrix: Matrix,
    workpiece: WorkPiece,
) -> Ops:
    """
    Processes a single artifact and returns its transformed ops.

    Args:
        artifact: The workpiece artifact.
        world_matrix: The world transformation matrix.
        workpiece: The workpiece.

    Returns:
        The transformed ops.
    """
    ops = artifact.ops.copy()
    _apply_artifact_scaling(ops, artifact, workpiece)

    tx, ty, angle, sx, sy, skew = world_matrix.decompose()
    workpiece_placement_matrix = _create_workpiece_placement_matrix(
        tx, ty, angle, sy, skew
    )
    ops.transform(workpiece_placement_matrix.to_4x4_numpy())

    return ops


def _apply_transformers_to_ops(
    ops: Ops,
    transformers: List["OpsTransformer"],
    context: Optional[ProgressContext] = None,
    stock_geometries: Optional[List["Geometry"]] = None,
) -> None:
    """
    Applies transformers to ops.

    Args:
        ops: The ops to transform.
        transformers: List of transformers to apply.
        context: Optional progress context.
        stock_geometries: List of stock boundary geometries in world space.
    """
    for i, transformer in enumerate(transformers):
        base_progress = 0.5 + (i / len(transformers) * 0.4)
        set_progress(
            context,
            base_progress,
            _("Applying '{t}'").format(t=transformer.label),
        )
        transformer.run(
            ops,
            workpiece=None,
            context=context,
            stock_geometries=stock_geometries,
        )


def compute_step_artifacts(
    artifacts: List[Tuple[WorkPieceArtifact, Matrix, WorkPiece]],
    transformers: List["OpsTransformer"],
    generation_id: int,
    context: Optional[ProgressContext] = None,
    stock_geometries: Optional[List["Geometry"]] = None,
) -> StepOpsArtifact:
    """
    Computes step ops artifact from workpiece artifacts and transforms.

    Args:
        artifacts: List of tuples containing (WorkPieceArtifact, world_matrix,
                   workpiece) for each workpiece in the assembly.
        transformers: List of OpsTransformers to apply to the combined ops.
        generation_id: The generation ID for staleness checking.
        context: Optional ProgressContext for progress reporting.
        stock_geometries: List of stock boundary geometries in world space.

    Returns:
        A StepOpsArtifact containing the combined and transformed operations.
    """
    combined_ops = Ops()
    num_items = len(artifacts)

    for i, (artifact, world_matrix, workpiece) in enumerate(artifacts):
        set_progress(context, i / num_items * 0.5)

        ops = _process_artifact(artifact, world_matrix, workpiece)
        combined_ops.workpiece_start(workpiece.uid)
        combined_ops.extend(ops)
        combined_ops.workpiece_end(workpiece.uid)

    _apply_transformers_to_ops(
        combined_ops,
        transformers,
        context,
        stock_geometries,
    )

    set_progress(context, 0.95)

    ops_artifact = StepOpsArtifact(
        ops=combined_ops,
        generation_id=generation_id,
    )

    set_progress(context, 1.0)

    return ops_artifact
