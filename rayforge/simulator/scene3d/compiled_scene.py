from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ...pipeline.artifact.base import BaseArtifact
from ...pipeline.artifact.handle import BaseArtifactHandle

if TYPE_CHECKING:
    from raygeo.compressed_array import CompressedArray


@dataclass
class VertexLayer:
    powered_verts: CompressedArray
    powered_attrib: CompressedArray
    travel_verts: CompressedArray
    zero_power_verts: CompressedArray
    powered_cmd_offsets: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    travel_cmd_offsets: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    is_rotary: bool = False


@dataclass
class TextureLayer:
    power_texture: CompressedArray
    width_px: int
    height_px: int
    model_matrix: np.ndarray
    cylinder_vertices: np.ndarray | None = None
    rotary_diameter: float = 0.0
    rotary_enabled: bool = False
    activation_cmd_idx: int = -1
    laser_uid: str = ""


@dataclass
class ScanlineOverlayLayer:
    positions: CompressedArray
    overlay_attrib: CompressedArray
    cmd_offsets: np.ndarray
    is_rotary: bool = False


@dataclass
class StockLayer:
    """Compiled solid-stock prism for a visible stock item.

    Positions are in world (machine mm) coordinates with the top face
    on the z=0 engrave plane and the body extruded toward negative z;
    ``transform`` maps them into visual space at draw time.  UVs are
    ``world_xy / texture_size_mm`` so the texture tiles at a physical
    density and repeats via ``GL_REPEAT``.
    """

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    transform: np.ndarray
    texture_path: str | None = None
    texture_size_mm: float = 300.0
    roughness: float = 0.8
    metallic: float = 0.0
    is_rotary: bool = False
    fallback_rgba: tuple[float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )
    # Resolved per-instance tint color (RGBA), or None for no tint.
    # Applied on the GPU as colorization (luma * tint) in the stock shader.
    tint_rgba: tuple[float, float, float, float] | None = None


class CompiledSceneArtifactHandle(BaseArtifactHandle):
    def __init__(
        self,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: dict[str, Any] | None = None,
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
        vertex_layers: list[VertexLayer],
        texture_layers: list[TextureLayer],
        overlay_layers: list[ScanlineOverlayLayer],
        laser_uid_order: list[str] | None = None,
        stock_layers: list[StockLayer] | None = None,
    ):
        self.generation_id = generation_id
        self.vertex_layers = vertex_layers
        self.texture_layers = texture_layers
        self.overlay_layers = overlay_layers
        self.laser_uid_order = laser_uid_order or []
        self.stock_layers = stock_layers or []

    def build_handle(self, key: str) -> CompiledSceneArtifactHandle:
        return CompiledSceneArtifactHandle(
            key=key,
            handle_class_name=CompiledSceneArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
        )
