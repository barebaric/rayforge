import numpy as np

from rayforge.pipeline.artifact.store import ArtifactStore
from rayforge.ui_gtk.sim3d.scene3d.compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    TextureLayer,
    VertexLayer,
)


def _make_vertex_layer(n_powered=10, n_travel=5, n_zero=3):
    return VertexLayer(
        powered_verts=np.random.rand(n_powered, 3).astype(np.float32),
        power_values=np.random.rand(n_powered).astype(np.float32),
        laser_indices=np.zeros(n_powered, dtype=np.float32),
        travel_verts=np.random.rand(n_travel, 3).astype(np.float32),
        zero_power_verts=np.random.rand(n_zero, 3).astype(np.float32),
        powered_cmd_offsets=[0, 2, n_powered],
        travel_cmd_offsets=[0, n_travel],
    )


def _make_texture_layer(with_cylinder=True):
    kw = {}
    if with_cylinder:
        kw["cylinder_vertices"] = np.random.rand(64, 3).astype(np.float32)
    return TextureLayer(
        power_texture=np.random.randint(0, 255, (32, 48), dtype=np.uint8),
        width_px=48,
        height_px=32,
        model_matrix=np.eye(4, dtype=np.float32),
        **kw,
    )


def _make_overlay_layer(n_cmds=5, n_verts=20):
    return ScanlineOverlayLayer(
        positions=np.random.rand(n_verts, 3).astype(np.float32),
        power_values=np.random.rand(n_verts).astype(np.float32),
        laser_indices=np.zeros(n_verts, dtype=np.float32),
        cmd_offsets=list(range(0, n_verts + 1, n_verts // n_cmds)),
    )


def _roundtrip(artifact, store, tag):
    handle = store.put(artifact, creator_tag=tag)
    loaded = store.get(handle)
    assert isinstance(loaded, CompiledSceneArtifact)
    return handle, loaded


class TestCompiledSceneArtifactRoundTrip:
    def test_vertex_layers_only(self):
        artifact = CompiledSceneArtifact(
            generation_id=42,
            vertex_layers=[_make_vertex_layer(), _make_vertex_layer()],
            texture_layers=[],
            overlay_layers=[],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_vl")

        assert loaded.generation_id == 42
        assert len(loaded.vertex_layers) == 2
        assert len(loaded.texture_layers) == 0
        assert len(loaded.overlay_layers) == 0

        for orig, restored in zip(
            artifact.vertex_layers, loaded.vertex_layers
        ):
            np.testing.assert_array_equal(
                orig.powered_verts, restored.powered_verts
            )
            np.testing.assert_array_equal(
                orig.power_values, restored.power_values
            )
            np.testing.assert_array_equal(
                orig.laser_indices, restored.laser_indices
            )
            np.testing.assert_array_equal(
                orig.travel_verts, restored.travel_verts
            )
            np.testing.assert_array_equal(
                orig.zero_power_verts, restored.zero_power_verts
            )
            assert orig.powered_cmd_offsets == restored.powered_cmd_offsets
            assert orig.travel_cmd_offsets == restored.travel_cmd_offsets

        store.release(handle)

    def test_texture_layers_with_optionals(self):
        artifact = CompiledSceneArtifact(
            generation_id=1,
            vertex_layers=[],
            texture_layers=[
                _make_texture_layer(with_cylinder=True),
                _make_texture_layer(with_cylinder=False),
            ],
            overlay_layers=[],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_tl")

        assert len(loaded.texture_layers) == 2

        tl0 = loaded.texture_layers[0]
        assert tl0.width_px == 48
        assert tl0.height_px == 32
        assert tl0.cylinder_vertices is not None
        np.testing.assert_array_equal(
            artifact.texture_layers[0].power_texture, tl0.power_texture
        )

        tl1 = loaded.texture_layers[1]
        assert tl1.cylinder_vertices is None

        store.release(handle)

    def test_overlay_layers(self):
        overlay = _make_overlay_layer()
        artifact = CompiledSceneArtifact(
            generation_id=7,
            vertex_layers=[],
            texture_layers=[],
            overlay_layers=[overlay],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_ol")

        assert len(loaded.overlay_layers) == 1
        ol = loaded.overlay_layers[0]
        np.testing.assert_array_equal(overlay.positions, ol.positions)
        np.testing.assert_array_equal(overlay.power_values, ol.power_values)
        np.testing.assert_array_equal(overlay.laser_indices, ol.laser_indices)
        assert overlay.cmd_offsets == ol.cmd_offsets

        store.release(handle)

    def test_empty_artifact(self):
        artifact = CompiledSceneArtifact(
            generation_id=0,
            vertex_layers=[],
            texture_layers=[],
            overlay_layers=[],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_empty")

        assert loaded.generation_id == 0
        assert len(loaded.vertex_layers) == 0
        assert len(loaded.texture_layers) == 0
        assert len(loaded.overlay_layers) == 0

        store.release(handle)

    def test_empty_vertex_layer(self):
        vl = VertexLayer(
            powered_verts=np.empty((0, 3), dtype=np.float32),
            power_values=np.empty((0,), dtype=np.float32),
            laser_indices=np.empty((0,), dtype=np.float32),
            travel_verts=np.empty((0, 3), dtype=np.float32),
            zero_power_verts=np.empty((0, 3), dtype=np.float32),
            powered_cmd_offsets=[],
            travel_cmd_offsets=[],
        )
        artifact = CompiledSceneArtifact(
            generation_id=0,
            vertex_layers=[vl],
            texture_layers=[],
            overlay_layers=[],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_ev")

        assert loaded.vertex_layers[0].powered_verts.shape == (0, 3)
        assert loaded.vertex_layers[0].powered_cmd_offsets == []
        assert loaded.vertex_layers[0].travel_cmd_offsets == []

        store.release(handle)

    def test_large_arrays(self):
        big_vl = VertexLayer(
            powered_verts=np.random.rand(50000, 3).astype(np.float32),
            power_values=np.random.rand(50000).astype(np.float32),
            laser_indices=np.zeros(50000, dtype=np.float32),
            travel_verts=np.random.rand(10000, 3).astype(np.float32),
            zero_power_verts=np.empty((0, 3), dtype=np.float32),
        )
        artifact = CompiledSceneArtifact(
            generation_id=0,
            vertex_layers=[big_vl],
            texture_layers=[],
            overlay_layers=[],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_big")

        np.testing.assert_array_equal(
            big_vl.powered_verts, loaded.vertex_layers[0].powered_verts
        )

        store.release(handle)

    def test_multiple_layer_types(self):
        artifact = CompiledSceneArtifact(
            generation_id=99,
            vertex_layers=[_make_vertex_layer()],
            texture_layers=[_make_texture_layer()],
            overlay_layers=[_make_overlay_layer()],
        )
        store = ArtifactStore()
        handle, loaded = _roundtrip(artifact, store, "test_mix")

        assert loaded.generation_id == 99
        assert len(loaded.vertex_layers) == 1
        assert len(loaded.texture_layers) == 1
        assert len(loaded.overlay_layers) == 1

        store.release(handle)
