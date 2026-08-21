from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ...pipeline.artifact.base import BaseArtifact
from ...pipeline.artifact.handle import BaseArtifactHandle
from .picking import PickContext, PickMesh, SceneItem

if TYPE_CHECKING:
    from raygeo.compressed_array import CompressedArray

# Unit quad as two triangles on the z=0 engrave plane, matching the
# quad renderers' GL_TRIANGLE_FAN winding.
_QUAD_TRIANGLES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass
class VertexLayer(SceneItem):
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
class TextureLayer(SceneItem):
    power_texture: CompressedArray
    width_px: int
    height_px: int
    model_matrix: np.ndarray
    cylinder_vertices: np.ndarray | None = None
    rotary_diameter: float = 0.0
    rotary_enabled: bool = False
    activation_cmd_idx: int = -1
    laser_uid: str = ""

    def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
        if self.rotary_enabled:
            if self.cylinder_vertices is None or ctx.cyl_model is None:
                return None
            verts = np.asarray(
                self.cylinder_vertices, dtype=np.float32
            ).reshape(-1, 5)
            return PickMesh(verts[:, :3], ctx.cyl_model)
        return PickMesh(_QUAD_TRIANGLES, self.model_matrix)


@dataclass
class ScanlineOverlayLayer(SceneItem):
    positions: CompressedArray
    overlay_attrib: CompressedArray
    cmd_offsets: np.ndarray
    is_rotary: bool = False


@dataclass
class WorkpieceImage(SceneItem):
    """A workpiece base image drawn as a quad (or rotary cylinder wrap).

    ``pixels`` is an RGBA uint8 array; ``model_matrix`` maps the unit
    quad into world space.  Rotary workpieces carry
    ``cylinder_vertices`` (a pre-baked triangle mesh wrapping the image
    around the cylinder) and ``rotary_diameter``; those instances are
    placed by the current cylinder model matrix at pick time.
    """

    pixels: np.ndarray
    model_matrix: np.ndarray
    cylinder_vertices: np.ndarray | None = None
    rotary_diameter: float = 0.0

    def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
        if self.cylinder_vertices is not None:
            if ctx.cyl_model is None:
                return None
            verts = np.asarray(
                self.cylinder_vertices, dtype=np.float32
            ).reshape(-1, 5)
            return PickMesh(verts[:, :3], ctx.cyl_model)
        return PickMesh(_QUAD_TRIANGLES, self.model_matrix)


@dataclass
class StockLayer(SceneItem):
    """Compiled solid-stock prism for a visible stock item.

    Positions are in world (machine mm) coordinates with the top face
    on the z=0 engrave plane and the body extruded toward negative z;
    ``transform`` maps them into visual space at draw time.  UVs are
    ``world_xy / texture_size_mm`` so the texture tiles at a physical
    density and repeats via ``GL_REPEAT``.

    When the stock has a folded burn surface map, ``power_texture``
    carries it as an R8 :class:`CompressedArray` (bottom-up rows),
    ``power_size_px`` its ``(width, height)``, ``power_aabb`` the
    world-mm burn grid rect ``(min_x, min_y, max_x, max_y)``, and
    ``power_uvs`` the per-vertex coordinates into that grid sampled
    by the stock shader's burn pass.
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
    # Burn-in surface map (R8, world-y-up rows) and its placement.
    power_texture: CompressedArray | None = None
    power_size_px: tuple[int, int] | None = None
    power_aabb: tuple[float, float, float, float] | None = None
    power_uvs: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )

    def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
        matrix = ctx.cyl_model if self.is_rotary else self.transform
        if matrix is None:
            return None
        positions = np.asarray(self.positions, dtype=np.float32)
        indices = np.asarray(self.indices, dtype=np.int64)
        if indices.size == 0:
            return None
        return PickMesh(positions.reshape(-1, 3)[indices], matrix)


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
        burn_layer_indices: set[int] | None = None,
    ):
        self.generation_id = generation_id
        self.vertex_layers = vertex_layers
        self.texture_layers = texture_layers
        self.overlay_layers = overlay_layers
        self.laser_uid_order = laser_uid_order or []
        self.stock_layers = stock_layers or []
        # Indices into ``texture_layers`` whose engrave output is
        # burned into a visible stock's surface map; the texture
        # renderer skips them unless the LUT overlay debug toggle is
        # on.
        self.burn_layer_indices = burn_layer_indices or set()

    def build_handle(self, key: str) -> CompiledSceneArtifactHandle:
        return CompiledSceneArtifactHandle(
            key=key,
            handle_class_name=CompiledSceneArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
        )
