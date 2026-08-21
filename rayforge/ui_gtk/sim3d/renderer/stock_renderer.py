"""
Renders compiled solid stock layers with the PBR stock shader.

Texture decoding (pyvips) and index expansion run in the upload
worker thread (:func:`prepare_stock_layer`); GL resource creation
runs on the GL thread.  Albedo textures are cached by
``(path, mtime, size)`` so scene rebuilds re-use the existing GL
texture objects instead of re-uploading multi-megabyte WebPs.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvips
from OpenGL import GL
from raygeo.image.pbr import generate_brdf_lut

from ....simulator.scene3d import CompiledSceneArtifact, StockLayer
from ..gl_utils import ShaderSet
from ..render_context import RenderContext
from .base import BaseRenderer

logger = logging.getLogger(__name__)

# Size of the split-sum BRDF integration LUT.
BRDF_LUT_SIZE = 32

# Decoded material textures to keep in the CPU-side cache.
MAX_CACHED_DECODES = 8

# Offset applied to the flat stock fill so coplanar engrave quads win
# the depth test without visible z-fighting.
_STOCK_POLYGON_OFFSET = (1.0, 1.0)

# Rotary stock: units-only offset.  The slope-scaled term grows with
# the surface's per-pixel depth slope, which is unbounded on the
# edge-on end caps and near the lateral silhouette — pushing those
# fragments so deep that geometry behind the rod wins the depth test
# and shows through the solid.  The units term alone is a sub-mm
# nudge, just enough to keep the coincident engrave cylinder from
# z-fighting.
_STOCK_ROTARY_POLYGON_OFFSET = (0.0, 1.0)


@dataclass
class PreparedStockLayer:
    """Stock layer data ready for GL upload, built off the main thread."""

    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    transform: np.ndarray
    roughness: float
    metallic: float
    fallback_rgba: tuple[float, float, float, float]
    tint_rgba: tuple[float, float, float, float] | None = None
    texture_key: tuple[str, int, int] | None = None
    texture_pixels: np.ndarray | None = None
    is_rotary: bool = False
    # Burn-in surface map (R8, world-y-up rows) decoded for GL upload,
    # its size, and the per-vertex coordinates into it.
    power_pixels: np.ndarray | None = None
    power_size: tuple[int, int] | None = None
    power_uvs: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )


@lru_cache(maxsize=MAX_CACHED_DECODES)
def _decode_texture_cached(
    path_str: str, mtime_ns: int, size: int
) -> tuple[int, int, np.ndarray] | None:
    """Decode a material texture into GL-ready RGBA pixels.

    The pixels are vertically flipped so texture row 0 is the bottom
    row, matching the v-up UV mapping of the stock mesh.  The file
    identity (path, mtime, size) is part of the cache key so replaced
    files are picked up automatically.
    """
    try:
        image = pyvips.Image.new_from_file(path_str).colourspace("srgb")
        if not image.hasalpha():
            image = image.addalpha()
        image = image.cast("uchar")
        height, width = image.height, image.width
        pixels = np.frombuffer(
            image.write_to_memory(), dtype=np.uint8
        ).reshape(height, width, 4)
        return width, height, np.ascontiguousarray(pixels[::-1])
    except (pyvips.error.Error, OSError):
        logger.warning("Failed to decode material texture: %s", path_str)
        return None


def _texture_key(
    texture_path: str | None,
) -> tuple[str, int, int] | None:
    """Cache key for a texture path, or None when unkeyable/missing."""
    if not texture_path:
        return None
    path = Path(texture_path)
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def prepare_stock_layer(layer: StockLayer) -> PreparedStockLayer:
    """Expands indices and decodes the textures without touching GL.

    Runs in a worker thread; the result is uploaded with GL calls on
    the main thread.
    """
    indices = layer.indices.astype(np.int64)
    positions = layer.positions.reshape(-1, 3)[indices]
    normals = layer.normals.reshape(-1, 3)[indices]
    uvs = layer.uvs.reshape(-1, 2)[indices]
    power_uvs = (
        layer.power_uvs.reshape(-1, 2)[indices]
        if layer.power_uvs.size
        else np.empty((0, 2), dtype=np.float32)
    )

    key = _texture_key(layer.texture_path)
    pixels: np.ndarray | None = None
    if key is not None:
        decoded = _decode_texture_cached(*key)
        if decoded is not None:
            _width, _height, pixels = decoded
        # Tinting happens on the GPU (per-instance shader uniform); the
        # decoded texture is shared and cached unchanged.

    power_pixels: np.ndarray | None = None
    power_size: tuple[int, int] | None = None
    if layer.power_texture is not None and layer.power_size_px is not None:
        decoded_burn = layer.power_texture.to_numpy()
        if decoded_burn is not None and decoded_burn.size:
            power_pixels = np.ascontiguousarray(decoded_burn)
            power_size = layer.power_size_px

    return PreparedStockLayer(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
        transform=layer.transform,
        roughness=layer.roughness,
        metallic=layer.metallic,
        fallback_rgba=layer.fallback_rgba,
        tint_rgba=layer.tint_rgba,
        texture_key=key if pixels is not None else None,
        texture_pixels=pixels,
        is_rotary=layer.is_rotary,
        power_pixels=power_pixels,
        power_size=power_size,
        power_uvs=np.ascontiguousarray(power_uvs, dtype=np.float32),
    )


class StockRenderer(BaseRenderer):
    """Renders solid stock meshes (flat prisms, rotary cylinder shells).

    Meshes arrive as :class:`StockLayer` objects on the compiled
    scene artifact; each layer becomes one draw instance with its own
    VAO/VBOs.  Flat instances draw with the bed-anchored world->visual
    transform; rotary instances draw with the per-frame cylinder
    kinematics (spinning with the chuck during playback).  Albedo
    textures live in a ``(path, mtime, size)`` keyed cache that
    survives scene rebuilds, so only new or changed materials trigger
    a decode and GL upload.
    """

    visibility_key = "show_stock"

    def __init__(self):
        super().__init__()
        self.is_initialized = False
        self.instances: list[dict] = []
        self._texture_cache: dict[tuple[str, int, int], int] = {}
        self._brdf_lut_texture: int = 0
        self._mvp_ui: np.ndarray | None = None
        self._camera_pos: np.ndarray | None = None
        self._point_light_pos: np.ndarray | None = None
        self._cyl_mvp: np.ndarray | None = None
        self._cyl_model: np.ndarray | None = None

    def prepare(self, ctx: RenderContext) -> None:
        """Caches the per-frame matrices and light positions."""
        self._mvp_ui = ctx.camera.mvp_ui
        self._camera_pos = ctx.camera.camera_position
        self._point_light_pos = ctx.kinematics.laser_light_pos
        self._cyl_mvp = ctx.kinematics.cylinder_mesh_mvp()
        self._cyl_model = ctx.kinematics.cylinder_model_matrix()

    def init_gl(self) -> None:
        """Creates the BRDF LUT texture used by the IBL split-sum."""
        if self.is_initialized:
            return
        lut = generate_brdf_lut(BRDF_LUT_SIZE).astype(np.float16)
        self._brdf_lut_texture = self._create_texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._brdf_lut_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 2)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RG16F,
            lut.shape[1],
            lut.shape[0],
            0,
            GL.GL_RG,
            GL.GL_HALF_FLOAT,
            lut,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.is_initialized = True
        logger.debug("StockRenderer initialized")

    def _create_gl_texture(self, pixels: np.ndarray) -> int:
        """Uploads RGBA pixels as an sRGB texture with mipmaps."""
        texture_id = GL.glGenTextures(1)
        height, width = pixels.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_SRGB8_ALPHA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id

    def _create_power_texture(self, pixels: np.ndarray) -> int:
        """Uploads an R8 burn power map.

        The buffer is world-y-up (row 0 = min y) and the mesh's
        ``power_uvs`` v grows with y, so rows upload as-is. No
        mipmaps: the burn signal is low-frequency and crisp edges are
        wanted; switch to a mipmapped filter if aliasing appears at
        grazing angles.
        """
        texture_id = GL.glGenTextures(1)
        height, width = pixels.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R8,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id

    def _create_instance_buffers(
        self, prepared: PreparedStockLayer
    ) -> tuple[int, list[int]]:
        """Creates the VAO/VBOs for one prepared stock layer."""
        vao = GL.glGenVertexArrays(1)
        vbo_pos = GL.glGenBuffers(1)
        vbo_norm = GL.glGenBuffers(1)
        vbo_uv = GL.glGenBuffers(1)
        vbos = [vbo_pos, vbo_norm, vbo_uv]

        attributes = [
            (0, vbo_pos, prepared.positions),
            (1, vbo_norm, prepared.normals),
            (2, vbo_uv, prepared.uvs),
        ]
        if prepared.power_uvs.size:
            vbo_power = GL.glGenBuffers(1)
            vbos.append(vbo_power)
            attributes.append((3, vbo_power, prepared.power_uvs))

        GL.glBindVertexArray(vao)
        for attr, vbo, data in attributes:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW
            )
            size = data.shape[1]
            GL.glVertexAttribPointer(
                attr, size, GL.GL_FLOAT, GL.GL_FALSE, 0, None
            )
            GL.glEnableVertexAttribArray(attr)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        return vao, vbos

    def _delete_instance(self, instance: dict) -> None:
        """Deletes one instance's VAO/VBOs and burn texture."""
        try:
            GL.glDeleteVertexArrays(1, [instance["vao"]])
            GL.glDeleteBuffers(len(instance["vbos"]), instance["vbos"])
            power_texture_id = instance.get("power_texture_id")
            if power_texture_id:
                GL.glDeleteTextures([power_texture_id])
        except GL.GLError:
            logger.exception("Error deleting stock instance buffers")

    def clear(self) -> None:
        """Drops all instances and their GL resources (full teardown)."""
        if not self.is_initialized:
            return
        for instance in self.instances:
            self._delete_instance(instance)
        self.instances.clear()
        if self._texture_cache:
            GL.glDeleteTextures(list(self._texture_cache.values()))
            self._texture_cache.clear()

    def upload_prepared(self, prepared_layers: list[PreparedStockLayer]):
        """Rebuilds all instances from prepared layers.

        GL textures for materials that are no longer referenced are
        deleted; unchanged ``(path, mtime, size)`` keys keep theirs.
        """
        if not self.is_initialized:
            return
        for instance in self.instances:
            self._delete_instance(instance)
        self.instances.clear()

        needed_keys = {p.texture_key for p in prepared_layers if p.texture_key}
        for key in list(self._texture_cache):
            if key not in needed_keys:
                GL.glDeleteTextures([self._texture_cache.pop(key)])

        for prepared in prepared_layers:
            texture_id = 0
            if prepared.texture_key is not None:
                texture_id = self._texture_cache.get(prepared.texture_key)
                if texture_id is None and prepared.texture_pixels is not None:
                    texture_id = self._create_gl_texture(
                        prepared.texture_pixels
                    )
                    self._texture_cache[prepared.texture_key] = texture_id
            power_texture_id = 0
            if (
                prepared.power_pixels is not None
                and prepared.power_size is not None
                and prepared.power_uvs.size
            ):
                # Burn maps change with every fold, so they are not
                # cached; they are a few hundred KB at the stock-grid
                # budget.
                power_texture_id = self._create_power_texture(
                    prepared.power_pixels
                )
                logger.info(
                    "Stock renderer: uploaded burn texture %sx%s px (id %s)",
                    prepared.power_size[0],
                    prepared.power_size[1],
                    power_texture_id,
                )
            vao, vbos = self._create_instance_buffers(prepared)
            self.instances.append(
                {
                    "vao": vao,
                    "vbos": vbos,
                    "vertex_count": len(prepared.positions),
                    "transform": prepared.transform,
                    "roughness": prepared.roughness,
                    "metallic": prepared.metallic,
                    "fallback_rgba": prepared.fallback_rgba,
                    "tint_rgba": prepared.tint_rgba,
                    "texture_id": texture_id,
                    "is_rotary": prepared.is_rotary,
                    "power_texture_id": power_texture_id,
                }
            )

    def update_from_artifact(self, artifact: CompiledSceneArtifact):
        """Prepares and uploads the artifact's stock layers synchronously."""
        prepared = [
            prepare_stock_layer(layer) for layer in artifact.stock_layers
        ]
        self.upload_prepared(prepared)

    def render(self, ctx: RenderContext, shaders: ShaderSet, **kwargs):
        """Draws every stock instance through the PBR stock shader."""
        if not self.is_initialized or not self.instances:
            return
        shader = shaders.stock
        if shader is None or self._mvp_ui is None:
            return

        shader.use()
        shader.set_vec3("uLightDir", (0.5, 0.8, 1.0))
        shader.set_vec3("uLightDir2", (-0.6, -0.4, 0.3))
        if self._camera_pos is not None:
            shader.set_vec3("uCameraPos", self._camera_pos)
        if self._point_light_pos is not None:
            shader.set_vec3("uPointLightPos", self._point_light_pos)
            shader.set_float("uPointLightOn", 1.0)
        else:
            shader.set_float("uPointLightOn", 0.0)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uBrdfLut", 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._brdf_lut_texture)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        shader.set_int("uPowerTexture", 2)

        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # Push the stock fill slightly away from the camera so the
        # engrave quads on the coplanar top face win the depth test.
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)

        try:
            for instance in self.instances:
                GL.glPolygonOffset(
                    *(
                        _STOCK_ROTARY_POLYGON_OFFSET
                        if instance.get("is_rotary")
                        else _STOCK_POLYGON_OFFSET
                    )
                )
                self._draw_instance(shader, instance)
        finally:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

    def _draw_instance(self, shader, instance: dict) -> None:
        if instance.get("is_rotary"):
            if self._cyl_mvp is None or self._cyl_model is None:
                return
            mvp = self._cyl_mvp
            model = self._cyl_model
        else:
            mvp = self._mvp_ui @ instance["transform"]
            model = instance["transform"]
        shader.set_mat4("uMVP", mvp)
        shader.set_mat4("uModel", model)
        shader.set_vec4("uAlbedo", instance["fallback_rgba"])
        shader.set_float("uRoughness", instance["roughness"])
        shader.set_float("uMetallic", instance["metallic"])
        shader.set_float("uAlpha", 1.0)

        texture_id = instance["texture_id"]
        shader.set_float("uUseTexture", 1.0 if texture_id else 0.0)
        tint = instance.get("tint_rgba")
        if tint is not None:
            shader.set_vec3("uTint", (tint[0], tint[1], tint[2]))
            shader.set_float("uUseTint", 1.0)
        else:
            shader.set_vec3("uTint", (1.0, 1.0, 1.0))
            shader.set_float("uUseTint", 0.0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id if texture_id else 0)

        power_texture_id = instance.get("power_texture_id", 0)
        shader.set_float("uUsePowerTexture", 1.0 if power_texture_id else 0.0)
        if power_texture_id:
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, power_texture_id)

        GL.glBindVertexArray(instance["vao"])
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, instance["vertex_count"])
        GL.glBindVertexArray(0)
