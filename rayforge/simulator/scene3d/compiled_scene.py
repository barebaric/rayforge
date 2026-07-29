from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...pipeline.artifact.base import BaseArtifact
from ...pipeline.artifact.handle import BaseArtifactHandle


@dataclass
class VertexLayer:
    powered_verts: np.ndarray
    power_values: np.ndarray
    laser_indices: np.ndarray
    travel_verts: np.ndarray
    zero_power_verts: np.ndarray
    powered_cmd_offsets: Any = field(default_factory=list)
    travel_cmd_offsets: Any = field(default_factory=list)
    is_rotary: bool = False


@dataclass
class TextureLayer:
    power_texture: np.ndarray
    width_px: int
    height_px: int
    model_matrix: np.ndarray
    cylinder_vertices: Optional[np.ndarray] = None
    rotary_diameter: float = 0.0
    rotary_enabled: bool = False
    activation_cmd_idx: int = -1
    laser_uid: str = ""


@dataclass
class ScanlineOverlayLayer:
    positions: np.ndarray
    power_values: np.ndarray
    laser_indices: np.ndarray
    cmd_offsets: Any
    is_rotary: bool = False


class CompiledSceneArtifactHandle(BaseArtifactHandle):
    def __init__(
        self,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            key=key,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )


class CompiledSceneArtifact(BaseArtifact):
    def __init__(
        self,
        generation_id: int,
        vertex_layers: List[VertexLayer],
        texture_layers: List[TextureLayer],
        overlay_layers: List[ScanlineOverlayLayer],
        laser_uid_order: Optional[List[str]] = None,
    ):
        self.generation_id = generation_id
        self.vertex_layers = vertex_layers
        self.texture_layers = texture_layers
        self.overlay_layers = overlay_layers
        self.laser_uid_order = laser_uid_order or []

    def build_handle(self, key: str) -> CompiledSceneArtifactHandle:
        return CompiledSceneArtifactHandle(
            key=key,
            handle_class_name=CompiledSceneArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
        )
