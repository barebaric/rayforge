"""
CPU ray-casting helpers for picking points on 3D scene geometry.

The 3D canvas orbits around the point under the cursor.  Instead of
always projecting onto the floor plane, the cursor ray is cast against
the actual scene geometry so the orbit pivot lands on the object.

The pick primitives (:class:`PickMesh`, :class:`PickScene`,
:class:`SceneItem`, :func:`build_pick_scene`) live in the simulator
layer so scene items can implement them without depending on the UI.
This module keeps only the camera-side math: unprojecting the cursor
ray and intersecting it with a pick scene.
"""

from __future__ import annotations

import numpy as np

from ...simulator.scene3d.picking import PickScene
from .camera import Camera


def camera_ray(
    camera: Camera, x: float, y: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Unprojects the cursor into a ``(origin, direction)`` ray.

    Two clip points are unprojected so the ray is correct for both
    perspective (converging) and orthographic (parallel) projections.
    Returns ``None`` when the ray cannot be constructed.
    """
    ndc_x = (2.0 * x) / camera.width - 1.0
    ndc_y = 1.0 - (2.0 * y) / camera.height

    try:
        inv_proj = np.linalg.inv(camera.get_projection_matrix())
        inv_view = np.linalg.inv(camera.get_view_matrix())
    except np.linalg.LinAlgError:
        return None

    near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    far_clip = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)
    near_eye = inv_proj @ near_clip
    far_eye = inv_proj @ far_clip
    near_world = inv_view @ (near_eye / near_eye[3])
    far_world = inv_view @ (far_eye / far_eye[3])

    ray_dir = far_world[:3] - near_world[:3]
    norm = np.linalg.norm(ray_dir)
    if norm < 1e-6:
        return None
    return near_world[:3], ray_dir / norm


def _ray_triangle_mesh(
    origin: np.ndarray, direction: np.ndarray, positions: np.ndarray
) -> float | None:
    """Nearest hit distance of a ray against a triangle soup, or None."""
    n_tri = len(positions) // 3
    if n_tri == 0:
        return None
    tri = positions[: n_tri * 3].reshape(n_tri, 3, 3)
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.sum(edge1 * h, axis=1)
    denom_ok = np.abs(a) > 1e-9
    f = np.zeros_like(a)
    np.divide(1.0, a, out=f, where=denom_ok)

    s = origin - v0
    u = f * np.sum(s * h, axis=1)
    q = np.cross(s, edge1)
    v = f * np.sum(direction * q, axis=1)
    t = f * np.sum(edge2 * q, axis=1)

    hit = denom_ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    ts = t[hit]
    if ts.size == 0:
        return None
    return float(ts.min())


def pick_point(
    scene: PickScene, camera: Camera, x: float, y: float
) -> np.ndarray | None:
    """Returns the nearest visual-space point hit by the cursor ray.

    Returns ``None`` when nothing in the scene is hit.
    """
    ray = camera_ray(camera, x, y)
    if ray is None:
        return None
    origin, direction = ray
    best_t = np.inf
    for mesh in scene.meshes:
        t = _ray_triangle_mesh(origin, direction, mesh.positions)
        if t is not None and t < best_t:
            best_t = t
    if best_t == np.inf:
        return None
    return origin + best_t * direction
