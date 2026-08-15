from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import trimesh

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.render_context import (
    CameraContext,
    KinematicsContext,
    RenderContext,
    ViewportContext,
)
from rayforge.ui_gtk.sim3d.renderer.model_renderer import (
    ModelRenderer,
    _load_mesh_data,
    _model_cache,
)


def _make_simple_mesh():
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )


class TestLoadMeshData:
    def setup_method(self):
        _model_cache.clear()

    def test_returns_none_on_missing_file(self):
        result = _load_mesh_data(Path("/nonexistent/file.glb"))
        assert result is None

    def test_caches_result(self, tmp_path):
        glb_file = tmp_path / "test.glb"
        glb_file.write_bytes(b"fake")

        mesh = _make_simple_mesh()

        with patch("trimesh.load", return_value=mesh):
            result = _load_mesh_data(glb_file)

        assert result is not None
        assert result.triangle_count == 1
        assert glb_file in _model_cache

        with patch("trimesh.load") as mock_load:
            result2 = _load_mesh_data(glb_file)
            mock_load.assert_not_called()

        assert result2 is result

    def test_handles_scene_with_geometry(self, tmp_path):
        glb_file = tmp_path / "multi.glb"
        glb_file.write_bytes(b"fake")

        mesh = _make_simple_mesh()
        scene = trimesh.Scene(geometry={"mesh1": mesh})

        with patch("trimesh.load", return_value=scene):
            result = _load_mesh_data(glb_file)

        assert result is not None
        assert result.triangle_count == 1

    def test_returns_none_on_trimesh_error(self, tmp_path):
        glb_file = tmp_path / "bad.glb"
        glb_file.write_bytes(b"fake")

        with patch("trimesh.load", side_effect=Exception("parse error")):
            result = _load_mesh_data(glb_file)

        assert result is None


class TestModelRenderer:
    def setup_method(self):
        _model_cache.clear()

    def test_init_defaults(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb")
        assert renderer._vao == 0
        assert renderer._vertex_count == 0
        assert renderer._bounds is None
        assert renderer._loaded is False

    def test_load_mesh_failure_returns_false(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "nonexistent.glb")
        assert renderer._load_mesh() is False
        assert renderer._loaded is False

    def test_load_mesh_success(self, tmp_path):
        glb_file = tmp_path / "test.glb"
        glb_file.write_bytes(b"fake")

        mesh = _make_simple_mesh()

        with patch("trimesh.load", return_value=mesh):
            renderer = ModelRenderer(glb_file)
            result = renderer._load_mesh()

        assert result is True
        assert renderer._loaded is True
        assert renderer._vertex_count == 3
        assert renderer._bounds is not None

    def test_render_noop_without_vao(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb")
        mock_shader = MagicMock()
        mvp = np.eye(4, dtype=np.float32)
        ctx = RenderContext(
            camera=CameraContext(
                mvp_ui=mvp,
                viewport_height=800,
                camera_position=np.zeros(3),
                color_set=ColorSet(),
            ),
            viewport=ViewportContext(),
            kinematics=KinematicsContext(mvp_ui=mvp),
        )
        renderer.render(ctx, mock_shader)
        mock_shader.use.assert_not_called()

    def test_bounds_property(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb")
        assert renderer.bounds is None

    def test_prepare_reads_kinematics_from_context(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb", link_name="gantry")
        mvp = np.eye(4, dtype=np.float32)
        kinematics = KinematicsContext(
            mvp_ui=mvp,
            model_world_transforms={"gantry": np.eye(4, dtype=np.float32)},
        )
        kinematics.laser_light_pos = np.array([1.0, 2.0, 3.0])
        ctx = RenderContext(
            camera=CameraContext(
                mvp_ui=mvp,
                viewport_height=800,
                camera_position=np.zeros(3),
                color_set=ColorSet(),
            ),
            viewport=ViewportContext(margin_shift=np.eye(4)),
            kinematics=kinematics,
        )

        renderer.prepare(ctx)

        assert renderer._mvp_matrix is not None
        assert renderer._model_matrix is not None
        assert renderer._point_light_pos is not None
        np.testing.assert_allclose(renderer._point_light_pos, [1.0, 2.0, 3.0])

    def test_prepare_projects_native_model_into_panel(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb", link_name="gantry")
        module_transform = np.eye(4, dtype=np.float32)
        module_transform[0, 3] = 10.0
        world_to_panel = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 100.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        kinematics = KinematicsContext(
            mvp_ui=np.eye(4, dtype=np.float32),
            model_world_transforms={"gantry": module_transform},
        )
        ctx = RenderContext(
            camera=CameraContext(
                mvp_ui=np.eye(4, dtype=np.float32),
                viewport_height=800,
                camera_position=np.zeros(3),
                color_set=ColorSet(),
            ),
            viewport=ViewportContext(
                world_to_panel=world_to_panel,
                margin_shift=np.eye(4, dtype=np.float32),
            ),
            kinematics=kinematics,
        )

        renderer.prepare(ctx)

        expected = world_to_panel @ module_transform
        assert renderer._model_matrix is not None
        assert renderer._mvp_matrix is not None
        np.testing.assert_allclose(renderer._model_matrix, expected)
        np.testing.assert_allclose(renderer._mvp_matrix, expected)

    def test_prepare_noop_without_kinematics(self, tmp_path):
        renderer = ModelRenderer(tmp_path / "test.glb", link_name="gantry")
        mvp = np.eye(4, dtype=np.float32)
        ctx = RenderContext(
            camera=CameraContext(
                mvp_ui=mvp,
                viewport_height=800,
                camera_position=np.zeros(3),
                color_set=ColorSet(),
            ),
            viewport=ViewportContext(margin_shift=np.eye(4)),
            kinematics=KinematicsContext(mvp_ui=mvp),
        )
        renderer.prepare(ctx)
        assert renderer._mvp_matrix is None
