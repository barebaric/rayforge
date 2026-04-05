"""
Renders a .glb 3D model using OpenGL triangles with per-vertex normals.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from OpenGL import GL

from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


@dataclass
class _CachedModelData:
    positions: np.ndarray
    normals: np.ndarray
    colors: Optional[np.ndarray]
    faces: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    triangle_count: int


_model_cache: Dict[Path, _CachedModelData] = {}


def _extract_color(mesh) -> Optional[np.ndarray]:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    try:
        vc = visual.vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            return np.array(vc, dtype=np.float32) / 255.0
    except Exception:
        pass
    try:
        mat = visual.material
        base = getattr(mat, "baseColorFactor", None)
        if base is None:
            base = getattr(mat, "diffuse", None)
        if base is not None:
            c = np.array(base, dtype=np.float32)
            if c.max() > 1.0:
                c = c / 255.0
            if c.shape[0] == 3:
                c = np.append(c, 1.0)
            return np.tile(c, (len(mesh.vertices), 1))
    except Exception:
        pass
    return None


def _load_mesh_data(path: Path) -> Optional[_CachedModelData]:
    cached = _model_cache.get(path)
    if cached is not None:
        return cached

    try:
        import trimesh

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
        bounds = (
            mesh.bounds[0].copy(),
            mesh.bounds[1].copy(),
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
    except Exception as e:
        logger.error("Failed to load model %s: %s", path, e)
        return None


def get_model_extent(path: Path) -> Optional[float]:
    data = _load_mesh_data(path)
    if data is None:
        return None
    bmin, bmax = data.bounds
    return float(np.max(bmax - bmin))


class ModelRenderer(BaseRenderer):
    """Loads and renders a .glb model as GL_TRIANGLES."""

    def __init__(self, resolved_path: Path):
        super().__init__()
        self._path = resolved_path
        self._vao: int = 0
        self._vbo_pos: int = 0
        self._vbo_norm: int = 0
        self._vbo_color: int = 0
        self._vertex_count: int = 0
        self._has_colors: bool = False
        self._bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._loaded: bool = False
        self._mesh_data: Optional[_CachedModelData] = None

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
        if not self._loaded:
            if not self._load_mesh():
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

    def render(
        self,
        shader: Shader,
        mvp_matrix: np.ndarray,
        model_matrix: Optional[np.ndarray] = None,
        camera_position: Optional[np.ndarray] = None,
    ) -> None:
        if not self._vao:
            return

        light_dir = np.array([0.5, 0.8, 1.0], dtype=np.float32)

        if model_matrix is not None and camera_position is not None:
            model_inv = np.linalg.inv(model_matrix)
            cam_pos = model_inv[:3, :3] @ camera_position + model_inv[:3, 3]
            cam_pos = cam_pos.astype(np.float32)
        else:
            cam_pos = np.zeros(3, dtype=np.float32)

        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_float("uUseVertexColor", 1.0 if self._has_colors else 0.0)
        shader.set_vec4("uColor", (0.5, 0.6, 0.7, 1.0))
        shader.set_float("uHasNormals", 1.0)
        shader.set_vec3("uLightDir", light_dir)
        shader.set_vec3("uCameraPos", cam_pos)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)
        GL.glBindVertexArray(0)

    @property
    def bounds(self):
        return self._bounds
