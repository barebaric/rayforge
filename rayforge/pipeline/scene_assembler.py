"""
Defines the SceneAssembler, which creates a lightweight description of a
scene for rendering, avoiding the creation of a monolithic Ops object for
the UI.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from .artifact.base import TextureData
from .artifact.handle import BaseArtifactHandle

if TYPE_CHECKING:
    from .generator import OpsGenerator
    from ..core.doc import Doc


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
    Assembles a lightweight description of the scene for 3D rendering.

    This function iterates through all visible steps, and for each one, it
    retrieves a handle to its cached `StepArtifact`. This artifact contains
    the final, aggregated, and transformed geometry for the entire step,
    ready for rendering.

    Args:
        doc: The document containing all layers, workflows, and workpieces.
        ops_generator: The generator instance holding the artifact cache.

    Returns:
        A SceneDescription object containing render items for each step.
    """
    render_items: List[RenderItem] = []
    visible_steps = set()

    for layer in doc.layers:
        if layer.visible and layer.workflow:
            for step in layer.workflow.steps:
                visible_steps.add(step)

    for step in visible_steps:
        handle = ops_generator.get_step_artifact_handle(step.uid)
        if handle:
            item = RenderItem(
                artifact_handle=handle,
                texture_data=None,  # Loaded from artifact in the render thread
                world_transform=np.identity(4, dtype=np.float32),
                workpiece_size=(0.0, 0.0),  # Not applicable at step level
                step_uid=step.uid,
                workpiece_uid="",  # Not applicable at step level
            )
            render_items.append(item)

    return SceneDescription(render_items=render_items)
