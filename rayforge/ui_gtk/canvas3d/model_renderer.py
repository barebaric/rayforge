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
    faces: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    triangle_count: int


_model_cache: Dict[Path, _CachedModelData] = {}


def _load_mesh_data(path: Path) -> Optional[_CachedModelData]:
    cached = _model_cache.get(path)
    if cached is not None:
        return cached

    try:
        import trimesh

        loaded = trimesh.load(str(path), file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            meshes = []
            for node in loaded.graph.nodes_geometry:
                transform, geom_name = loaded.graph.get(node)
                geom = loaded.geometry[geom_name]
                geom = geom.apply_transform(transform)
                meshes.append(geom)
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
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
        self._vertex_count: int = 0
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
        self._loaded = True
        return True

    def init_gl(self) -> None:
        if not self._loaded:
            if not self._load_mesh():
                return

        self._vao = self._create_vao()
        self._vbo_pos = self._create_vbo()
        self._vbo_norm = self._create_vbo()

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
        shader.set_float("uUseVertexColor", 0.0)
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
