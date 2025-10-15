"""
Defines the SceneAssembler, which creates a lightweight description of a
scene for rendering, avoiding the creation of a monolithic Ops object for
the UI.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING, Dict
import numpy as np

from ..core.layer import Layer
from .artifact.base import TextureData
from .artifact.workpiece import WorkPieceArtifact
from .artifact.handle import BaseArtifactHandle

if TYPE_CHECKING:
    from .generator import OpsGenerator
    from ..core.doc import Doc
    from ..core.workpiece import WorkPiece
    from ..core.step import Step


logger = logging.getLogger(__name__)


@dataclass
class RenderItem:
    """A lightweight instruction for rendering one artifact."""

    artifact_handle: Optional[BaseArtifactHandle]
    texture_data: Optional[TextureData]
    world_transform: np.ndarray  # 4x4 numpy matrix
    workpiece_size: Tuple[float, float]
    step_uid: str
    workpiece_uid: str


@dataclass
class SceneDescription:
    """A complete, lightweight description of a scene for rendering."""

    render_items: List[RenderItem]


def generate_scene_description(
    doc: "Doc", ops_generator: "OpsGenerator"
) -> SceneDescription:
    """
    Assembles a lightweight description of the scene for rendering.

    This function iterates through all visible items, calculates their final
    world transformation matrix, and pairs it with a handle to the cached,
    untransformed artifact data. This avoids processing or concatenating large
    Ops objects on the main thread.

    Args:
        doc: The document containing all layers, workflows, and workpieces.
        ops_generator: The generator instance holding the artifact cache.

    Returns:
        A SceneDescription object.
    """
    render_items: List[RenderItem] = []

    # This logic is similar to the start of the old `generate_job_ops`, but
    # it only gathers instructions, it does not process data.
    work_items_by_layer: Dict[Layer, List[tuple[Step, WorkPiece]]] = {}
    for layer in doc.layers:
        renderable_items = layer.get_renderable_items()
        if renderable_items:
            work_items_by_layer[layer] = renderable_items

    for layer, items in work_items_by_layer.items():
        for step, workpiece in items:
            # Fetch both the handle and the full artifact. The handle is for
            # the vector part, while the full artifact is needed if it's a
            # raster type. `get_artifact` is a cached lookup, so it's fast.
            handle = ops_generator.get_artifact_handle(step.uid, workpiece.uid)
            artifact = ops_generator.get_artifact(step, workpiece)

            texture_data = None
            if isinstance(artifact, WorkPieceArtifact):
                texture_data = artifact.texture_data

            item = RenderItem(
                artifact_handle=handle,
                texture_data=texture_data,
                world_transform=workpiece.get_world_transform().to_4x4_numpy(),
                workpiece_size=workpiece.size,
                step_uid=step.uid,
                workpiece_uid=workpiece.uid,
            )
            render_items.append(item)

    return SceneDescription(render_items=render_items)
