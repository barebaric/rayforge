import pytest
import numpy as np
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.pipeline.scene_assembler import (
    RenderItem,
    SceneDescription,
    generate_scene_description,
)
from rayforge.pipeline.artifact import (
    StepArtifactHandle,
)


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
def mock_doc() -> Doc:
    """Create a mock document with layers."""
    doc = MagicMock(spec=Doc)
    doc.layers = []
    return doc


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline."""
    pipeline = MagicMock()
    pipeline.get_step_artifact_handle = MagicMock()
    return pipeline


@pytest.fixture
def mock_layer() -> Layer:
    """Create a mock layer with a workflow."""
    layer = MagicMock(spec=Layer)
    layer.name = "test_layer"
    layer.visible = True
    layer.workflow = MagicMock()
    layer.workflow.steps = []
    return layer


@pytest.fixture
def mock_step() -> Step:
    """Create a mock step."""
    step = MagicMock(spec=Step)
    step.uid = "test_step_uid"
    return step


@pytest.fixture
def mock_step_artifact_handle() -> StepArtifactHandle:
    """Create a mock StepArtifact handle."""
    return StepArtifactHandle(
        shm_name="test_step_shm",
        handle_class_name="StepArtifactHandle",
        artifact_type_name="StepArtifact",
        is_scalable=False,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=None,
        time_estimate=45.67,
    )


class TestRenderItem:
    """Test the RenderItem dataclass for the new Step-based structure."""

    def test_render_item_creation(self, mock_step_artifact_handle):
        """Test creating a RenderItem for a StepArtifact."""
        world_transform = np.eye(4)
        item = RenderItem(
            artifact_handle=mock_step_artifact_handle,
            texture_data=None,  # Texture is loaded later
            world_transform=world_transform,
            workpiece_size=(0.0, 0.0),  # Not applicable
            step_uid="step_1",
            workpiece_uid="",  # Not applicable
        )

        assert item.artifact_handle == mock_step_artifact_handle
        assert item.texture_data is None
        assert np.array_equal(item.world_transform, world_transform)
        assert item.workpiece_size == (0.0, 0.0)
        assert item.step_uid == "step_1"
        assert item.workpiece_uid == ""


class TestSceneDescription:
    """Test the SceneDescription dataclass."""

    def test_scene_description_creation(self, mock_step_artifact_handle):
        """Test creating a SceneDescription with render items."""
        world_transform = np.eye(4)
        items = [
            RenderItem(
                artifact_handle=mock_step_artifact_handle,
                texture_data=None,
                world_transform=world_transform,
                workpiece_size=(0.0, 0.0),
                step_uid="step_1",
                workpiece_uid="",
            ),
        ]

        scene = SceneDescription(render_items=items)
        assert len(scene.render_items) == 1
        assert scene.render_items[0].step_uid == "step_1"

    def test_scene_description_empty(self):
        """Test creating an empty SceneDescription."""
        scene = SceneDescription(render_items=[])
        assert len(scene.render_items) == 0


class TestGenerateSceneDescription:
    """Test the generate_scene_description function."""

    def test_empty_document(self, mock_doc, mock_pipeline):
        """Test with an empty document."""
        mock_doc.layers = []
        scene = generate_scene_description(mock_doc, mock_pipeline)
        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 0

    def test_invisible_layer(
        self, mock_doc, mock_pipeline, mock_layer, mock_step
    ):
        """Test that items from an invisible layer are ignored."""
        mock_layer.visible = False
        mock_layer.workflow.steps = [mock_step]
        mock_doc.layers = [mock_layer]

        scene = generate_scene_description(mock_doc, mock_pipeline)
        assert len(scene.render_items) == 0
        mock_pipeline.get_step_artifact_handle.assert_not_called()

    def test_step_with_cached_artifact(
        self,
        mock_doc,
        mock_pipeline,
        mock_layer,
        mock_step,
        mock_step_artifact_handle,
    ):
        """Test that a visible step with an artifact creates a RenderItem."""
        mock_layer.workflow.steps = [mock_step]
        mock_doc.layers = [mock_layer]
        mock_pipeline.get_step_artifact_handle.return_value = (
            mock_step_artifact_handle
        )

        scene = generate_scene_description(mock_doc, mock_pipeline)

        assert isinstance(scene, SceneDescription)
        assert len(scene.render_items) == 1

        item = scene.render_items[0]
        assert item.artifact_handle == mock_step_artifact_handle
        assert item.texture_data is None
        assert item.step_uid == mock_step.uid
        assert item.workpiece_uid == ""
        assert item.workpiece_size == (0.0, 0.0)
        assert np.array_equal(item.world_transform, np.eye(4))

        mock_pipeline.get_step_artifact_handle.assert_called_once_with(
            mock_step.uid
        )

    def test_step_without_cached_artifact(
        self, mock_doc, mock_pipeline, mock_layer, mock_step
    ):
        """Test that a step without a cached artifact is ignored."""
        mock_layer.workflow.steps = [mock_step]
        mock_doc.layers = [mock_layer]
        mock_pipeline.get_step_artifact_handle.return_value = None

        scene = generate_scene_description(mock_doc, mock_pipeline)

        assert len(scene.render_items) == 0
        mock_pipeline.get_step_artifact_handle.assert_called_once_with(
            mock_step.uid
        )

    def test_multiple_steps_and_layers(
        self, mock_doc, mock_pipeline, mock_step_artifact_handle
    ):
        """Test with a complex document structure."""
        # Layer 1: Visible, 2 steps, one with artifact, one without
        step1a = MagicMock(spec=Step, uid="s1a")
        step1b = MagicMock(spec=Step, uid="s1b")
        layer1 = MagicMock(spec=Layer, visible=True)
        layer1.workflow = MagicMock(steps=[step1a, step1b])

        # Layer 2: Invisible, should be ignored
        step2 = MagicMock(spec=Step, uid="s2")
        layer2 = MagicMock(spec=Layer, visible=False)
        layer2.workflow = MagicMock(steps=[step2])

        # Layer 3: Visible, 1 step with artifact
        step3 = MagicMock(spec=Step, uid="s3")
        layer3 = MagicMock(spec=Layer, visible=True)
        layer3.workflow = MagicMock(steps=[step3])

        mock_doc.layers = [layer1, layer2, layer3]

        # Mock cache returns
        def mock_get_handle(step_uid):
            if step_uid in ["s1a", "s3"]:
                return mock_step_artifact_handle
            return None

        mock_pipeline.get_step_artifact_handle.side_effect = mock_get_handle

        scene = generate_scene_description(mock_doc, mock_pipeline)

        # Should get 2 items: s1a and s3.
        # s1b is skipped (no artifact), s2 is skipped (invisible layer).
        assert len(scene.render_items) == 2
        uids = {item.step_uid for item in scene.render_items}
        assert uids == {"s1a", "s3"}

        assert mock_pipeline.get_step_artifact_handle.call_count == 3
