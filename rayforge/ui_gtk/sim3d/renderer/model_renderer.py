"""
Renders a .glb 3D model using OpenGL triangles with per-vertex normals.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from OpenGL import GL
from trimesh.visual.color import ColorVisuals
from trimesh.visual.material import PBRMaterial

from ....simulator.scene3d.picking import PickContext, PickMesh, SceneItem
from ..gl_utils import ShaderSet
from ..render_context import RenderContext
from .base import BaseRenderer

logger = logging.getLogger(__name__)


@dataclass
class _CachedModelData:
    positions: np.ndarray
    normals: np.ndarray
    colors: np.ndarray | None
    faces: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    triangle_count: int


_model_cache: dict[Path, _CachedModelData] = {}


def _extract_color(mesh: trimesh.Trimesh) -> np.ndarray | None:
    if mesh.visual is None:
        return None
    if isinstance(mesh.visual, ColorVisuals):
        vc = mesh.visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            return np.array(vc, dtype=np.float32) / 255.0
        return None
    mat = mesh.visual.material
    if isinstance(mat, PBRMaterial):
        base = mat.baseColorFactor
    else:
        base = mat.diffuse
    if base is not None:
        c = np.array(base, dtype=np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        if c.shape[0] == 3:
            c = np.append(c, 1.0)
        return np.tile(c, (len(mesh.vertices), 1))
    return None


def _load_mesh_data(path: Path) -> _CachedModelData | None:
    cached = _model_cache.get(path)
    if cached is not None:
        return cached

    try:
        loaded = trimesh.load(str(path), file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            meshes = []
            colors = []
            for node in loaded.graph.nodes_geometry:
                transform, geom_name = loaded.graph.get(node)
                geom = loaded.geometry[geom_name]
                color = _extract_color(geom)
                geom = geom.apply_transform(transform)
                meshes.append(geom)
                if color is not None:
                    colors.append(color)
            mesh = trimesh.util.concatenate(meshes)
            assert isinstance(mesh, trimesh.Trimesh)
            has_colors = len(colors) == len(meshes) and sum(
                c.shape[0] for c in colors
            ) == len(mesh.vertices)
            vertex_colors = (
                np.vstack(colors).astype(np.float32) if has_colors else None
            )
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
            vertex_colors = _extract_color(mesh)
        else:
            logger.error(
                "Unexpected type from trimesh.load: %s",
                type(loaded).__name__,
            )
            return None

        assert isinstance(mesh, trimesh.Trimesh)

        positions = np.array(mesh.vertices, dtype=np.float32)
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

        y_up_to_z_up = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
        )
        positions = (y_up_to_z_up @ positions.T).T
        normals = (y_up_to_z_up @ normals.T).T

        bounds = (
            positions.min(axis=0),
            positions.max(axis=0),
        )

        faces = np.array(mesh.faces, dtype=np.uint32)
        triangle_count = len(faces)

        data = _CachedModelData(
            positions=positions,
            normals=normals,
            colors=vertex_colors,
            faces=faces,
            bounds=bounds,
            triangle_count=triangle_count,
        )
        _model_cache[path] = data
        return data
    except Exception as e:  # noqa: BLE001 - trimesh library boundary
        logger.error("Failed to load model %s: %s", path, e)
        return None


def get_model_extent(path: Path) -> float | None:
    data = _load_mesh_data(path)
    if data is None:
        return None
    bmin, bmax = data.bounds
    return float(np.max(bmax - bmin))


def model_world_matrix(
    link_name: str,
    kinematics,
    viewport,
) -> np.ndarray | None:
    """Per-frame model matrix mapping a model into visual space.

    Uses the link's world transform from the kinematics, applying the
    focused rotary head position when active, and maps it into the
    visual frame through the viewport's panel transform.
    """
    if not kinematics.model_world_transforms:
        return None
    module_transform = kinematics.model_world_transforms.get(link_name)
    if module_transform is None:
        return None
    module_transform = module_transform.astype(np.float32)
    if kinematics.is_rotary:
        focused = kinematics.focused_rotary_head_positions
        if focused and link_name in focused:
            module_transform[:3, 3] = focused[link_name].astype(np.float32)
    physical_to_visual = viewport.margin_shift @ viewport.world_to_panel
    return (physical_to_visual @ module_transform).astype(np.float32)


def model_triangle_positions(path: Path) -> np.ndarray | None:
    """Triangle-expanded vertex positions of a loaded model, or None."""
    data = _load_mesh_data(path)
    if data is None:
        return None
    return data.positions[data.faces.flatten()]


@dataclass
class MachineModel(SceneItem):
    """A machine-assembly link's 3D model as a pickable scene item.

    The geometry lives in the UI-side GLB cache; the current transform
    is carried per-frame by :class:`PickContext` so the mesh is
    translated to the object's current position at pick time, exactly
    like every other scene item.
    """

    path: Path
    link_name: str

    def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
        positions = model_triangle_positions(self.path)
        if positions is None or len(positions) == 0:
            return None
        matrix = ctx.model_matrices.get(self.link_name)
        if matrix is None:
            return None
        return PickMesh(positions, matrix)


class ModelRenderer(BaseRenderer):
    """Loads and renders a .glb model as GL_TRIANGLES."""

    visibility_key = "show_models"

    def __init__(self, model: MachineModel):
        super().__init__()
        self._model = model
        self._path = model.path
        self.link_name = model.link_name
        self._vao: int = 0
        self._vbo_pos: int = 0
        self._vbo_norm: int = 0
        self._vbo_color: int = 0
        self._vertex_count: int = 0
        self._has_colors: bool = False
        self._bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._loaded: bool = False
        self._mesh_data: _CachedModelData | None = None
        self._mvp_matrix: np.ndarray | None = None
        self._model_matrix: np.ndarray | None = None
        self._point_light_pos: np.ndarray | None = None

    def prepare(self, ctx: RenderContext) -> None:
        """Computes and caches the per-frame matrices for the model mesh."""
        self._point_light_pos = ctx.kinematics.laser_light_pos
        if ctx.viewport is None:
            return
        model_matrix = model_world_matrix(
            self.link_name, ctx.kinematics, ctx.viewport
        )
        self._model_matrix = model_matrix
        if model_matrix is None:
            self._mvp_matrix = None
            return
        self._mvp_matrix = ctx.camera.mvp_ui @ model_matrix

    def _load_mesh(self) -> bool:
        self._mesh_data = _load_mesh_data(self._path)
        if self._mesh_data is None:
            return False

        flat_indices = self._mesh_data.faces.flatten()
        self._positions = self._mesh_data.positions[flat_indices]
        self._normals = self._mesh_data.normals[flat_indices]
        self._vertex_count = len(flat_indices)
        self._bounds = self._mesh_data.bounds
        if self._mesh_data.colors is not None:
            self._colors = self._mesh_data.colors[flat_indices]
            self._has_colors = True
        self._loaded = True
        return True

    def init_gl(self) -> None:
        if not self._loaded and not self._load_mesh():
            return

        self._vao = self._create_vao()
        self._vbo_pos = self._create_vbo()
        self._vbo_norm = self._create_vbo()
        if self._has_colors:
            self._vbo_color = self._create_vbo()

        GL.glBindVertexArray(self._vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_pos)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._positions.nbytes,
            self._positions,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        if self._has_colors:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_color)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                self._colors.nbytes,
                self._colors,
                GL.GL_STATIC_DRAW,
            )
            GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_norm)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._normals.nbytes,
            self._normals,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_TRUE, 0, None)
        GL.glEnableVertexAttribArray(2)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(self, ctx: RenderContext, shaders: ShaderSet, **kwargs) -> None:
        if not self._vao or self._mvp_matrix is None:
            return

        shader = shaders.main
        if shader is None:
            return

        light_dir = np.array([0.5, 0.8, 1.0], dtype=np.float32)
        fill_dir = np.array([-0.6, -0.4, 0.3], dtype=np.float32)
        camera_position = ctx.camera.camera_position
        model_matrix = self._model_matrix
        point_light_pos = self._point_light_pos

        if model_matrix is not None and camera_position is not None:
            model_inv = np.linalg.inv(model_matrix)
            cam_pos = model_inv[:3, :3] @ camera_position + model_inv[:3, 3]
            cam_pos = cam_pos.astype(np.float32)
            if point_light_pos is not None:
                point_light_pos = (
                    model_inv[:3, :3] @ point_light_pos + model_inv[:3, 3]
                )
                point_light_pos = point_light_pos.astype(np.float32)
        else:
            cam_pos = np.zeros(3, dtype=np.float32)

        shader.use()
        shader.set_mat4("uMVP", self._mvp_matrix)
        shader.set_float("uUseVertexColor", 1.0 if self._has_colors else 0.0)
        shader.set_vec4("uColor", (0.5, 0.6, 0.7, 1.0))
        shader.set_float("uHasNormals", 1.0)
        shader.set_vec3("uLightDir", light_dir)
        shader.set_vec3("uLightDir2", fill_dir)
        shader.set_vec3("uCameraPos", cam_pos)
        if point_light_pos is not None:
            shader.set_vec3("uPointLightPos", point_light_pos)
            shader.set_float("uPointLightOn", 1.0)
        else:
            shader.set_vec3("uPointLightPos", np.zeros(3, dtype=np.float32))
            shader.set_float("uPointLightOn", 0.0)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)

    @property
    def bounds(self):
        return self._bounds
