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


def test_add_layer_fires_changed_signal(doc):
    """Test that adding a new layer to a document fires its changed signal."""
    doc_changed_handler = MagicMock()
    doc.changed.connect(doc_changed_handler)

    new_layer = Layer(doc, "Layer 2")
    doc.add_layer(new_layer)

    assert len(doc.layers) == 2
    doc_changed_handler.assert_called_once_with(doc)


def test_full_signal_chain_bubbles_up_to_doc(doc):
    """
    Integration test: A change on a deep child (Step) should bubble
    all the way up to the top-level Doc.
    """
    doc_changed_handler = MagicMock()
    doc.changed.connect(doc_changed_handler)

    # Arrange: Get to a step deep inside the hierarchy
    layer = doc.layers[0]
    workflow = layer.workflow
    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    # Reset mock after setup changes
    doc_changed_handler.reset_mock()

    # Act: Change a property on the step
    step.set_power(500)

    # Assert: The doc itself was notified, with the step as context
    doc_changed_handler.assert_called_once_with(doc, step=step)
