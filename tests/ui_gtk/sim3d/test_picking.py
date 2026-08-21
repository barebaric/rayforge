"""Tests for the 3D canvas cursor ray-casting helpers."""

import numpy as np
import pytest
from raygeo.compressed_array import CompressedArray

from rayforge.simulator.scene3d import (
    PickContext,
    StockLayer,
    TextureLayer,
    WorkpieceImage,
)
from rayforge.simulator.scene3d.picking import (
    PickMesh,
    PickScene,
    SceneItem,
    build_pick_scene,
)
from rayforge.ui_gtk.sim3d.camera import Camera, ViewDirection
from rayforge.ui_gtk.sim3d.picking import camera_ray, pick_point


def _camera() -> Camera:
    cam = Camera(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        640,
        480,
    )
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)
    return cam


def _quad(z: float = 0.0) -> np.ndarray:
    corners = np.array(
        [
            [0.0, 0.0, z],
            [100.0, 0.0, z],
            [100.0, 100.0, z],
            [0.0, 100.0, z],
        ],
        dtype=np.float32,
    )
    return np.vstack(
        [
            corners[0],
            corners[1],
            corners[2],
            corners[0],
            corners[2],
            corners[3],
        ]
    )


def test_camera_ray_points_down_from_top_view():
    cam = _camera()
    ray = camera_ray(cam, 320, 240)
    assert ray is not None
    origin, direction = ray
    assert direction[2] == pytest.approx(-1.0, abs=1e-6)
    assert direction[0] == pytest.approx(0.0, abs=1e-6)
    assert direction[1] == pytest.approx(0.0, abs=1e-6)
    assert origin[2] > 0.0


def test_pick_point_hits_mesh_above_plane():
    cam = _camera()
    scene = PickScene()
    scene.meshes.append(PickMesh(_quad(z=5.0)))

    point = pick_point(scene, cam, 320, 240)
    assert point is not None
    assert point[0] == pytest.approx(50.0, abs=1e-5)
    assert point[1] == pytest.approx(50.0, abs=1e-5)
    assert point[2] == pytest.approx(5.0, abs=1e-5)


def test_pick_point_returns_nearest_of_multiple_meshes():
    cam = _camera()
    scene = PickScene()
    scene.meshes.append(PickMesh(_quad(z=2.0)))
    scene.meshes.append(PickMesh(_quad(z=8.0)))

    point = pick_point(scene, cam, 320, 240)
    assert point is not None
    # Top-down camera: the higher surface is the nearer one.
    assert point[2] == pytest.approx(8.0, abs=1e-5)


def test_pick_point_miss_returns_none():
    cam = _camera()
    scene = PickScene()
    # A small quad off to the side, away from the center cursor ray.
    corners = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
        dtype=np.float32,
    )
    scene.meshes.append(PickMesh(corners))
    assert pick_point(scene, cam, 320, 240) is None


def test_pick_mesh_applies_matrix_transform():
    cam = _camera()
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    tri = np.vstack(
        [
            corners[0],
            corners[1],
            corners[2],
            corners[0],
            corners[2],
            corners[3],
        ]
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = 100.0
    matrix[1, 1] = 100.0
    matrix[2, 3] = 15.0

    scene = PickScene()
    scene.meshes.append(PickMesh(tri, matrix))
    point = pick_point(scene, cam, 320, 240)
    assert point is not None
    assert point[0] == pytest.approx(50.0, abs=1e-4)
    assert point[1] == pytest.approx(50.0, abs=1e-4)
    assert point[2] == pytest.approx(15.0, abs=1e-4)


def test_build_pick_scene_flat_stock_hits_top_face():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 10.0],
            [100.0, 0.0, 10.0],
            [100.0, 100.0, 10.0],
            [0.0, 100.0, 10.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([4, 5, 6, 4, 6, 7], dtype=np.uint32)
    layer = StockLayer(
        positions=positions,
        normals=np.zeros_like(positions),
        uvs=np.zeros((8, 2), dtype=np.float32),
        indices=indices,
        transform=np.eye(4, dtype=np.float32),
    )

    scene = build_pick_scene([layer], PickContext())
    assert scene is not None
    point = pick_point(scene, _camera(), 320, 240)
    assert point is not None
    assert point[2] == pytest.approx(10.0, abs=1e-5)


def test_build_pick_scene_applies_stock_transform():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = 100.0
    transform[1, 1] = 100.0
    transform[2, 3] = 20.0
    layer = StockLayer(
        positions=positions,
        normals=np.zeros_like(positions),
        uvs=np.zeros((4, 2), dtype=np.float32),
        indices=indices,
        transform=transform,
    )

    scene = build_pick_scene([layer], PickContext())
    assert scene is not None
    point = pick_point(scene, _camera(), 320, 240)
    assert point is not None
    assert point[0] == pytest.approx(50.0, abs=1e-4)
    assert point[1] == pytest.approx(50.0, abs=1e-4)
    assert point[2] == pytest.approx(20.0, abs=1e-4)


def test_build_pick_scene_empty_returns_none():
    assert build_pick_scene([], PickContext()) is None


def test_build_pick_scene_rotary_stock_skipped_without_cyl_model():
    positions = np.array(
        [
            [0.0, -5.0, 0.0],
            [10.0, -5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [10.0, -5.0, 0.0],
            [10.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    layer = StockLayer(
        positions=positions,
        normals=np.zeros_like(positions),
        uvs=np.zeros((6, 2), dtype=np.float32),
        indices=np.arange(6, dtype=np.uint32),
        transform=np.eye(4, dtype=np.float32),
        is_rotary=True,
    )

    assert build_pick_scene([layer], PickContext(cyl_model=None)) is None

    cyl_model = np.eye(4, dtype=np.float32)
    scene = build_pick_scene([layer], PickContext(cyl_model=cyl_model))
    assert scene is not None
    assert len(scene.meshes) == 1


def test_build_pick_scene_accepts_custom_scene_item():
    cam = _camera()

    class _PlaneItem(SceneItem):
        """A pickable scene item above the floor plane."""

        def pick_mesh(self, ctx: PickContext) -> PickMesh | None:
            return PickMesh(_quad(z=5.0))

    scene = build_pick_scene([_PlaneItem()], PickContext())
    assert scene is not None
    point = pick_point(scene, cam, 320, 240)
    assert point is not None
    assert point[2] == pytest.approx(5.0, abs=1e-5)


def test_texture_layer_pick_mesh_maps_quad():
    model_matrix = np.eye(4, dtype=np.float32)
    model_matrix[0, 0] = 100.0
    model_matrix[1, 1] = 100.0
    model_matrix[2, 3] = 12.0
    layer = TextureLayer(
        power_texture=CompressedArray.from_uint8_2d(
            np.zeros((2, 2), dtype=np.uint8)
        ),
        width_px=100,
        height_px=100,
        model_matrix=model_matrix,
    )

    scene = build_pick_scene([layer], PickContext())
    assert scene is not None
    point = pick_point(scene, _camera(), 320, 240)
    assert point is not None
    assert point[2] == pytest.approx(12.0, abs=1e-4)


def test_workpiece_image_pick_mesh_maps_quad():
    model_matrix = np.eye(4, dtype=np.float32)
    model_matrix[0, 0] = 100.0
    model_matrix[1, 1] = 100.0
    model_matrix[2, 3] = 7.0
    image = WorkpieceImage(
        pixels=np.zeros((2, 2, 4), dtype=np.uint8),
        model_matrix=model_matrix,
    )

    scene = build_pick_scene([image], PickContext())
    assert scene is not None
    point = pick_point(scene, _camera(), 320, 240)
    assert point is not None
    assert point[2] == pytest.approx(7.0, abs=1e-4)
