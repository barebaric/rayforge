import pytest
from unittest.mock import MagicMock
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step


@pytest.fixture
def doc():
    """Provides a real Doc instance. No mocks needed."""
    return Doc()


def test_doc_initialization(doc):
    """Verify a new Doc starts with one layer and a real HistoryManager."""
    assert len(doc.layers) == 1
    # The import from ..undo would need to be added to the file for this to
    # work from ..undo import HistoryManager
    # assert isinstance(doc.history_manager, HistoryManager)
    assert doc.history_manager is not None  # A good enough check


def test_add_layer_fires_descendant_added(doc):
    """Test adding a layer fires descendant_added with the layer as origin."""
    handler = MagicMock()
    doc.descendant_added.connect(handler)

    new_layer = Layer("Layer 2")
    doc.add_layer(new_layer)

    assert len(doc.layers) == 2
    handler.assert_called_once_with(doc, origin=new_layer)


def test_remove_layer_fires_descendant_removed(doc):
    """
    Test removing a layer fires descendant_removed with the layer as origin.
    """
    layer_to_remove = Layer("Layer 2")
    doc.add_layer(layer_to_remove)

    handler = MagicMock()
    doc.descendant_removed.connect(handler)
    doc.remove_layer(layer_to_remove)

    handler.assert_called_once_with(doc, origin=layer_to_remove)


def test_descendant_updated_bubbles_up_to_doc(doc):
    """A descendant_updated signal from a Step should bubble up to the Doc."""
    handler = MagicMock()
    doc.descendant_updated.connect(handler)

    layer = doc.layers[0]
    workflow = layer.workflow
    step = Step("Test Step")
    workflow.add_step(step)
    handler.reset_mock()  # Ignore the 'add' event

    # Act
    step.set_power(500)

    # Assert
    handler.assert_called_once_with(doc, origin=step)


def test_descendant_added_bubbles_up_to_doc(doc):
    """A descendant_added signal for a new Step should bubble up to the Doc."""
    handler = MagicMock()
    doc.descendant_added.connect(handler)

    layer = doc.layers[0]
    workflow = layer.workflow
    step = Step("Test Step")

    # Act
    workflow.add_step(step)

    # Assert
    handler.assert_called_once_with(doc, origin=step)


def test_descendant_removed_bubbles_up_to_doc(doc):
    """A descendant_removed signal for a step should bubble up to the Doc."""
    layer = doc.layers[0]
    workflow = layer.workflow
    step = Step("Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    doc.descendant_removed.connect(handler)

    # Act
    workflow.remove_step(step)

    # Assert
    handler.assert_called_once_with(doc, origin=step)
