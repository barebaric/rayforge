import pytest
import numpy as np
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step
from rayforge.core.matrix import Matrix
from rayforge.pipeline.scene_assembler import (
    RenderItem,
    SceneDescription,
    generate_scene_description,
)
from rayforge.pipeline.artifact.handle import ArtifactHandle
from rayforge.pipeline.artifact.base import Artifact, TextureData
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.core.ops import Ops


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """Setup test configuration for machine and laser."""
    from rayforge.machine.models.machine import Laser, Machine

    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.config.config", test_config)
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


@pytest.fixture
def mock_doc():
    """Create a mock document with layers."""
    doc = MagicMock(spec=Doc)
    doc.layers = []
    return doc


@pytest.fixture
def mock_ops_generator():
    """Create a mock ops generator with cache."""
    generator = MagicMock()
    generator._ops_cache = {}
    generator.get_artifact = MagicMock()
    return generator


@pytest.fixture
def mock_layer():
    """Create a mock layer with renderable items."""
    layer = MagicMock(spec=Layer)
    layer.name = "test_layer"
    layer.visible = True
    layer.get_renderable_items = MagicMock(return_value=[])
    return layer


@pytest.fixture
def mock_step():
    """Create a mock step."""
    step = MagicMock(spec=Step)
    step.uid = "test_step_uid"
    step.visible = True
    return step


@pytest.fixture
def mock_workpiece():
    """Create a mock workpiece."""
    workpiece = MagicMock(spec=WorkPiece)
    workpiece.uid = "test_workpiece_uid"
    workpiece.size = (10.0, 20.0)

    # Mock the get_world_transform method
    transform_matrix = MagicMock(spec=Matrix)
    transform_matrix.to_4x4_numpy.return_value = np.eye(4)
    workpiece.get_world_transform.return_value = transform_matrix

    return workpiece


@pytest.fixture
def mock_artifact_handle():
    """Create a mock artifact handle."""
    return ArtifactHandle(
        shm_name="test_shm",
        artifact_type="vector",
        is_scalable=True,
        source_coordinate_system_name="source",
        source_dimensions=(10.0, 20.0),
        generation_size=(10.0, 20.0),
    )


@pytest.fixture
def mock_hybrid_artifact():
    """Create a mock hybrid artifact."""
    texture_data = TextureData(
        power_texture_data=np.zeros((10, 10)),
        dimensions_mm=(10.0, 10.0),
        position_mm=(0.0, 0.0),
    )
    return Artifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        texture_data=texture_data,
    )


@pytest.fixture
def mock_vector_artifact():
    """Create a mock vector artifact."""
    return Artifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
    )


class TestRenderItem:
    """Test the RenderItem dataclass."""

    def test_render_item_creation(
        self, mock_artifact_handle, mock_hybrid_artifact
    ):
        """Test creating a RenderItem with all parameters."""
        world_transform = np.eye(4)
        item = RenderItem(
            artifact_handle=mock_artifact_handle,
            texture_data=mock_hybrid_artifact.texture_data,
            world_transform=world_transform,
            workpiece_size=(10.0, 20.0),
            step_uid="step_1",
            workpiece_uid="workpiece_1",
        )

        assert item.artifact_handle == mock_artifact_handle
        assert item.texture_data == mock_hybrid_artifact.texture_data
        assert np.array_equal(item.world_transform, world_transform)
        assert item.workpiece_size == (10.0, 20.0)
        assert item.step_uid == "step_1"
        assert item.workpiece_uid == "workpiece_1"

    def test_render_item_without_texture(self, mock_artifact_handle):
        """Test creating a RenderItem without texture data."""
        world_transform = np.eye(4)
        item = RenderItem(
            artifact_handle=mock_artifact_handle,
            texture_data=None,
            world_transform=world_transform,
            workpiece_size=(10.0, 20.0),
            step_uid="step_1",
            workpiece_uid="workpiece_1",
        )

        assert item.artifact_handle == mock_artifact_handle
        assert item.texture_data is None


class TestSceneDescription:
    """Test the SceneDescription dataclass."""

    def test_scene_description_creation(self, mock_artifact_handle):
        """Test creating a SceneDescription with render items."""
        world_transform = np.eye(4)
        items = [
            RenderItem(
                artifact_handle=mock_artifact_handle,
                texture_data=None,
                world_transform=world_transform,
                workpiece_size=(10.0, 20.0),
                step_uid="step_1",
                workpiece_uid="workpiece_1",
            ),
            RenderItem(
                artifact_handle=None,
                texture_data=None,
                world_transform=world_transform,
                workpiece_size=(15.0, 25.0),
                step_uid="step_2",
                workpiece_uid="workpiece_2",
            ),
        ]

        scene = SceneDescription(render_items=items)
        assert len(scene.render_items) == 2
        assert scene.render_items[0].step_uid == "step_1"
        assert scene.render_items[1].step_uid == "step_2"

    def test_scene_description_empty(self):
        """Test creating an empty SceneDescription."""
        scene = SceneDescription(render_items=[])
        assert len(scene.render_items) == 0


class TestGenerateSceneDescription:
    """Test the generate_scene_description function."""

    def test_empty_document(self, mock_doc, mock_ops_generator):
        """Test with an empty document."""
        mock_doc.layers = []

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 0

    def test_layer_without_renderable_items(
        self, mock_doc, mock_ops_generator, mock_layer
    ):
        """Test with a layer that has no renderable items."""
        mock_layer.get_renderable_items.return_value = []
        mock_doc.layers = [mock_layer]

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 0
        mock_layer.get_renderable_items.assert_called_once()

    def test_layer_with_renderable_items(
        self,
        mock_doc,
        mock_ops_generator,
        mock_layer,
        mock_step,
        mock_workpiece,
        mock_artifact_handle,
        mock_vector_artifact,
    ):
        """Test with a layer that has renderable items."""
        # Setup renderable items
        renderable_items = [(mock_step, mock_workpiece)]
        mock_layer.get_renderable_items.return_value = renderable_items
        mock_doc.layers = [mock_layer]

        # Setup artifact cache and generator
        key = (mock_step.uid, mock_workpiece.uid)
        mock_ops_generator._ops_cache[key] = mock_artifact_handle
        mock_ops_generator.get_artifact.return_value = mock_vector_artifact

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 1

        item = scene.render_items[0]
        assert item.artifact_handle == mock_artifact_handle
        assert item.texture_data is None
        assert item.step_uid == mock_step.uid
        assert item.workpiece_uid == mock_workpiece.uid
        assert item.workpiece_size == mock_workpiece.size
        assert np.array_equal(item.world_transform, np.eye(4))

        mock_ops_generator.get_artifact.assert_called_once_with(
            mock_step, mock_workpiece
        )

    def test_layer_with_hybrid_artifact(
        self,
        mock_doc,
        mock_ops_generator,
        mock_layer,
        mock_step,
        mock_workpiece,
        mock_artifact_handle,
        mock_hybrid_artifact,
    ):
        """Test with a layer that has a hybrid raster artifact."""
        # Setup renderable items
        renderable_items = [(mock_step, mock_workpiece)]
        mock_layer.get_renderable_items.return_value = renderable_items
        mock_doc.layers = [mock_layer]

        # Setup artifact cache and generator
        key = (mock_step.uid, mock_workpiece.uid)
        mock_ops_generator._ops_cache[key] = mock_artifact_handle
        mock_ops_generator.get_artifact.return_value = mock_hybrid_artifact

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 1

        item = scene.render_items[0]
        assert item.artifact_handle == mock_artifact_handle
        assert item.texture_data == mock_hybrid_artifact.texture_data
        assert item.step_uid == mock_step.uid
        assert item.workpiece_uid == mock_workpiece.uid

    def test_multiple_layers_with_items(
        self,
        mock_doc,
        mock_ops_generator,
        mock_step,
        mock_workpiece,
        mock_artifact_handle,
        mock_vector_artifact,
    ):
        """Test with multiple layers containing renderable items."""
        # Create two mock layers
        layer1 = MagicMock(spec=Layer)
        layer1.name = "layer1"
        layer1.get_renderable_items.return_value = [
            (mock_step, mock_workpiece)
        ]

        # Create second step and workpiece for layer2
        step2 = MagicMock(spec=Step)
        step2.uid = "step2_uid"
        workpiece2 = MagicMock(spec=WorkPiece)
        workpiece2.uid = "workpiece2_uid"
        workpiece2.size = (30.0, 40.0)
        transform_matrix = MagicMock(spec=Matrix)
        transform_matrix.to_4x4_numpy.return_value = np.eye(4)
        workpiece2.get_world_transform.return_value = transform_matrix

        layer2 = MagicMock(spec=Layer)
        layer2.name = "layer2"
        layer2.get_renderable_items.return_value = [(step2, workpiece2)]

        mock_doc.layers = [layer1, layer2]

        # Setup artifact cache and generator
        key1 = (mock_step.uid, mock_workpiece.uid)
        key2 = (step2.uid, workpiece2.uid)
        mock_ops_generator._ops_cache[key1] = mock_artifact_handle
        mock_ops_generator._ops_cache[key2] = mock_artifact_handle
        mock_ops_generator.get_artifact.return_value = mock_vector_artifact

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 2

        # Check first item
        item1 = scene.render_items[0]
        assert item1.step_uid == mock_step.uid
        assert item1.workpiece_uid == mock_workpiece.uid

        # Check second item
        item2 = scene.render_items[1]
        assert item2.step_uid == step2.uid
        assert item2.workpiece_uid == workpiece2.uid
        assert item2.workpiece_size == (30.0, 40.0)

    def test_missing_artifact_in_cache(
        self,
        mock_doc,
        mock_ops_generator,
        mock_layer,
        mock_step,
        mock_workpiece,
        mock_vector_artifact,
    ):
        """Test when artifact is not in cache."""
        # Setup renderable items
        renderable_items = [(mock_step, mock_workpiece)]
        mock_layer.get_renderable_items.return_value = renderable_items
        mock_doc.layers = [mock_layer]

        # Don't add to cache, but still return artifact from get_artifact
        mock_ops_generator.get_artifact.return_value = mock_vector_artifact

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 1

        item = scene.render_items[0]
        assert item.artifact_handle is None  # Should be None when not in cache
        assert item.texture_data is None  # Not a hybrid artifact
        assert item.step_uid == mock_step.uid
        assert item.workpiece_uid == mock_workpiece.uid

    def test_world_transform_conversion(
        self,
        mock_doc,
        mock_ops_generator,
        mock_layer,
        mock_step,
        mock_workpiece,
        mock_artifact_handle,
        mock_vector_artifact,
    ):
        """Test that world transform is properly converted to numpy array."""
        # Setup renderable items
        renderable_items = [(mock_step, mock_workpiece)]
        mock_layer.get_renderable_items.return_value = renderable_items
        mock_doc.layers = [mock_layer]

        # Setup artifact cache and generator
        key = (mock_step.uid, mock_workpiece.uid)
        mock_ops_generator._ops_cache[key] = mock_artifact_handle
        mock_ops_generator.get_artifact.return_value = mock_vector_artifact

        # Create a custom transformation matrix
        custom_transform = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        func = mock_workpiece.get_world_transform.return_value.to_4x4_numpy
        func.return_value = custom_transform

        scene = generate_scene_description(mock_doc, mock_ops_generator)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 1

        item = scene.render_items[0]
        assert np.array_equal(item.world_transform, custom_transform)
        func.assert_called_once()
