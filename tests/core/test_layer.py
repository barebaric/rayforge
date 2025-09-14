import pytest
from unittest.mock import MagicMock
from blinker import Signal
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step


@pytest.fixture
def doc():
    return Doc()


@pytest.fixture
def layer(doc):
    """Provides the active layer from a real Doc, which is a regular Layer."""
    return doc.active_layer


@pytest.fixture
def mock_workpiece_with_signals():
    """Provides a MagicMock of a WorkPiece with real Signal objects."""
    mock_wp = MagicMock(spec=WorkPiece)
    # DocItem base class expects these signals to exist for connection
    mock_wp.updated = Signal()
    mock_wp.transform_changed = Signal()
    mock_wp.descendant_added = Signal()
    mock_wp.descendant_removed = Signal()
    mock_wp.descendant_updated = Signal()
    mock_wp.descendant_transform_changed = Signal()
    mock_wp.parent = None
    mock_wp.name = "Mock WP"
    return mock_wp


def test_add_workpiece_fires_descendant_added(
    layer, mock_workpiece_with_signals
):
    """Adding a workpiece should fire descendant_added."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    layer.add_workpiece(mock_workpiece_with_signals)

    handler.assert_called_once_with(layer, origin=mock_workpiece_with_signals)
    assert mock_workpiece_with_signals in layer.workpieces


def test_workflow_descendant_added_bubbles_to_layer(layer):
    """A descendant_added signal from a workflow should bubble up."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    workflow = layer.workflow
    assert workflow is not None
    step = Step("Test Step")

    # Act
    workflow.add_step(step)

    # Assert
    handler.assert_called_once_with(layer, origin=step)


def test_workpiece_data_change_bubbles_up_to_layer(
    layer, mock_workpiece_with_signals
):
    """
    A data change on a workpiece (via .updated) should bubble a
    descendant_updated signal.
    """
    layer.add_workpiece(mock_workpiece_with_signals)

    descendant_updated_handler = MagicMock()
    layer.descendant_updated.connect(descendant_updated_handler)

    # Act: Simulate the .updated signal being fired from the child.
    mock_workpiece_with_signals.updated.send(mock_workpiece_with_signals)

    # Assert
    descendant_updated_handler.assert_called_once_with(
        layer, origin=mock_workpiece_with_signals
    )


def test_workpiece_transform_change_bubbles_up_to_layer(
    layer, mock_workpiece_with_signals
):
    """
    A transform change on a workpiece (via .transform_changed) should
    bubble a specific signal up.
    """
    layer.add_workpiece(mock_workpiece_with_signals)

    transform_changed_handler = MagicMock()
    layer.descendant_transform_changed.connect(transform_changed_handler)

    # Act: Simulate the .transform_changed signal firing.
    mock_workpiece_with_signals.transform_changed.send(
        mock_workpiece_with_signals
    )

    # Assert
    transform_changed_handler.assert_called_once_with(
        layer, origin=mock_workpiece_with_signals
    )
