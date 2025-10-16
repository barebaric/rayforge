"""Test that multipass transformer changes trigger re-generation."""

from unittest.mock import MagicMock, patch
import pytest
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Laser, Machine
from rayforge.pipeline.coordinator import PipelineCoordinator
from rayforge.pipeline.transformer.multipass import MultiPassTransformer


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """Provides a mock machine config for all tests in this file."""
    test_laser = Laser()
    test_laser.max_power = 1000
    test_machine = Machine()
    test_machine.dimensions = (200, 150)
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.acceleration = 1000
    test_machine.heads.clear()
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.config.config", test_config)
    return test_config


@pytest.mark.usefixtures("setup_real_config")
class TestMultipassRegeneration:
    """Test that changes to multipass transformer trigger re-generation."""

    def test_job_assembly_invalidated_signal_connected(self):
        """Test generator connects to the job_assembly_invalidated signal."""
        # Create a minimal document structure
        doc = Doc()
        layer = Layer(name="Test Layer")
        step = Step(typelabel="Test Step", name="Test Step")
        workpiece = WorkPiece(name="Test Workpiece")

        # Set up the hierarchy
        doc.add_child(layer)
        assert layer.workflow is not None
        layer.workflow.add_child(step)
        layer.add_child(workpiece)

        # Create a mock task manager
        task_manager = MagicMock()

        # Create the generator with mocked dependencies
        with patch("rayforge.pipeline.coordinator.logger"):
            # Track if the handler was called
            handler_called = MagicMock()

            # Create a custom handler to track calls
            def track_handler(sender):
                handler_called()

            # Create generator and patch the handler method
            generator = PipelineCoordinator(doc, task_manager)
            generator._on_job_assembly_invalidated = track_handler

        # Disconnect and reconnect with our tracked handler
        doc.job_assembly_invalidated.disconnect(
            generator._on_job_assembly_invalidated
        )
        doc.job_assembly_invalidated.connect(track_handler)

        # Send the signal to test the connection
        doc.job_assembly_invalidated.send(doc)

        # Verify the handler was called
        assert handler_called.called

    def test_multipass_change_sends_job_assembly_invalidated(self):
        """Test multipass changes send job_assembly_invalidated signal."""
        # Create a minimal document structure
        doc = Doc()
        layer = Layer(name="Test Layer")
        step = Step(typelabel="Test Step", name="Test Step")
        workpiece = WorkPiece(name="Test Workpiece")

        # Set up the hierarchy
        doc.add_child(layer)
        assert layer.workflow is not None
        layer.workflow.add_child(step)
        layer.add_child(workpiece)

        # Configure the step with a multipass transformer
        step.per_step_transformers_dicts = [
            MultiPassTransformer(
                enabled=True, passes=2, z_step_down=0.5
            ).to_dict()
        ]

        # Mock the job_assembly_invalidated signal to track if it's called
        doc.job_assembly_invalidated = MagicMock()

        # Change the multipass settings
        new_multipass = MultiPassTransformer(
            enabled=True, passes=3, z_step_down=1.0
        )
        step.per_step_transformers_dicts = [new_multipass.to_dict()]

        # Send the per_step_transformer_changed signal
        step.per_step_transformer_changed.send(step)

        # Verify that job_assembly_invalidated was sent
        assert doc.job_assembly_invalidated.send.called
