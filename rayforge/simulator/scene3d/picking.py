"""
Scene-item picking primitives shared by the simulator and the UI.

Every scene item derives from :class:`SceneItem` and may implement
:meth:`SceneItem.pick_mesh` to contribute pickable geometry to the
cursor ray-cast.  The picker only talks to the base class, so it never
needs to know the concrete scene item types.

This module is pure numpy and lives in the simulator layer so that
scene items defined there can return :class:`PickMesh` objects without
depending on the UI.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def _transform_points(
    points: np.ndarray, matrix: np.ndarray | None
) -> np.ndarray:
    """Applies a 4x4 matrix to (N,3) points, returning visual-space XYZ."""
    if matrix is None:
        return np.asarray(points, dtype=np.float32)
    n = len(points)
    ones = np.ones((n, 1), dtype=np.float32)
    homo = np.concatenate([np.asarray(points, dtype=np.float32), ones], axis=1)
    out = homo @ np.asarray(matrix, dtype=np.float32).T
    w = out[:, 3:4]
    safe_w = np.where(np.abs(w) < 1e-9, 1.0, w)
    return (out[:, :3] / safe_w).astype(np.float32)


class PickMesh:
    """A triangle soup in visual (camera) space, ready for ray tests.

    ``matrix`` optionally maps the raw positions into visual space with
    the same 4x4 transform the renderers apply at draw time.
    """

    def __init__(
        self, positions: np.ndarray, matrix: np.ndarray | None = None
    ):
        self.positions = _transform_points(positions, matrix)


class PickScene:
    """All pickable triangles in the current scene."""

    def __init__(self):
        self.meshes: list[PickMesh] = []


@dataclass
class PickContext:
    """Per-frame matrices the picker needs to place scene items.

    ``cyl_model`` is the current rotary cylinder model matrix, or
    ``None`` when no rotary frame is active; rotary items use it to
    place their geometry into visual space.  ``model_matrices`` carries
    each machine model link's current visual-space matrix keyed by link
    name, so :class:`MachineModel` items can be translated to their
    current position at pick time.
    """

    cyl_model: np.ndarray | None = None
    model_matrices: dict[str, np.ndarray] = field(default_factory=dict)


class SceneItem:
    """Base class for every scene item.

    Subclasses that own pickable geometry override :meth:`pick_mesh` to
    return a :class:`PickMesh` (geometry in visual space).  The default
    implementation returns ``None``, so non-pickable items simply
    inherit it.
    """

    def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
        return None


def build_pick_scene(
    items: Iterable[SceneItem], ctx: PickContext
) -> PickScene | None:
    """Collects the pick meshes of every scene item into a scene.

    Returns ``None`` when nothing contributes any geometry.
    """
    scene = PickScene()
    for item in items:
        mesh = item.pick_mesh(ctx)
        if mesh is not None and len(mesh.positions):
            scene.meshes.append(mesh)
    if not scene.meshes:
        return None
    return scene
