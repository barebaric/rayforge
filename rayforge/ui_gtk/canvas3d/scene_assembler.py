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
from ...pipeline.artifact.base import TextureData
from ...pipeline.artifact.handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ...pipeline.pipeline import Pipeline
    from ...core.doc import Doc


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
    doc: "Doc", pipeline: "Pipeline"
) -> SceneDescription:
    """
    Assembles a lightweight description of the scene for 3D rendering.

    This function iterates through all visible steps and creates a single
    RenderItem for each, pointing to the StepArtifact. The StepArtifact is a
    self-contained "render bundle" with all necessary data for the 3D canvas.
    """
    render_items: List[RenderItem] = []
    visible_steps = set()

    for layer in doc.layers:
        if layer.visible and layer.workflow:
            for step in layer.workflow.steps:
                visible_steps.add(step)

    for step in visible_steps:
        # Use the new lightweight render artifact for UI consumption
        handle = pipeline.get_step_render_artifact_handle(step.uid)
        if handle:
            item = RenderItem(
                artifact_handle=handle,
                texture_data=None,
                world_transform=np.identity(4, dtype=np.float32),
                workpiece_size=(0.0, 0.0),  # Not applicable at step level
                step_uid=step.uid,
                workpiece_uid="",  # Not applicable at step level
            )
            render_items.append(item)

    return SceneDescription(render_items=render_items)
