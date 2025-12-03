import pytest
from unittest.mock import MagicMock, patch
from blinker import Signal
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.layer import Layer
from rayforge.core.matrix import Matrix


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
    mock_wp.natural_size = (10.0, 10.0)
    mock_wp.get_local_bbox.return_value = (0.0, 0.0, 10.0, 10.0)
    mock_wp.matrix = Matrix.identity()
    return mock_wp


def test_add_workpiece_fires_descendant_added(
    layer, mock_workpiece_with_signals
):
    """Adding a workpiece should fire descendant_added."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    layer.add_workpiece(mock_workpiece_with_signals)

    handler.assert_called_once_with(
        layer, origin=mock_workpiece_with_signals, parent_of_origin=layer
    )
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
    handler.assert_called_once_with(
        layer, origin=step, parent_of_origin=workflow
    )


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
        layer, origin=mock_workpiece_with_signals, parent_of_origin=layer
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
        layer, origin=mock_workpiece_with_signals, parent_of_origin=layer
    )


def test_layer_stock_item_uid_property():
    """Tests that a Layer has a stock_item_uid property."""
    layer = Layer("Test Layer")

    # Default value should be None
    assert layer.stock_item_uid is None

    # Can set a value
    layer.stock_item_uid = "test-stock-uid"
    assert layer.stock_item_uid == "test-stock-uid"


def test_layer_stock_item_property():
    """Tests that a Layer can get and set stock items."""
    doc = Doc()
    layer = doc.active_layer
    stock_asset = StockAsset(name="Test Stock Asset")
    doc.add_asset(stock_asset)
    stock_item = StockItem(
        stock_asset_uid=stock_asset.uid, name="Test Stock Item"
    )
    doc.add_child(stock_item)

    # Initially no stock item assigned
    assert layer.stock_item is None

    # Assign stock item
    layer.stock_item = stock_item
    assert layer.stock_item is stock_item
    assert layer.stock_item_uid == stock_item.uid

    # Unassign stock item
    layer.stock_item = None
    assert layer.stock_item is None
    assert layer.stock_item_uid is None


def test_layer_stock_item_property_with_invalid_uid():
    """Tests that layer.stock_item returns None for invalid UID."""
    layer = Layer("Test Layer")

    # Set invalid UID
    layer.stock_item_uid = "non-existent-uid"

    # Should return None
    assert layer.stock_item is None


def test_layer_to_dict_includes_stock_item_uid():
    """Tests that to_dict includes the stock_item_uid property."""
    layer = Layer("Test Layer")
    layer.stock_item_uid = "test-stock-uid"

    data = layer.to_dict()

    assert "stock_item_uid" in data
    assert data["stock_item_uid"] == "test-stock-uid"


def test_layer_from_dict_handles_stock_item_uid():
    """Tests that from_dict handles the stock_item_uid property."""
    layer_dict = {
        "uid": "test-layer-uid",
        "type": "layer",
        "name": "Test Layer",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "visible": True,
        "stock_item_uid": "test-stock-uid",
        "children": [],
    }

    layer = Layer.from_dict(layer_dict)

    assert layer.stock_item_uid == "test-stock-uid"


def test_layer_from_dict_handles_missing_stock_item_uid():
    """Tests that from_dict handles missing stock_item_uid property."""
    layer_dict = {
        "uid": "test-layer-uid",
        "type": "layer",
        "name": "Test Layer",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "visible": True,
        "children": [],
    }

    layer = Layer.from_dict(layer_dict)

    assert layer.stock_item_uid is None


def test_layer_to_dict_serialization():
    """Tests serializing a Layer to a dictionary."""
    layer = Layer("Test Layer")
    layer.matrix = Matrix.translation(5, 10) @ Matrix.scale(2, 3)
    layer.visible = False
    layer.stock_item_uid = "test-stock-uid"

    wp = WorkPiece("test.svg")
    wp.matrix = Matrix.translation(1, 2)
    layer.add_child(wp)

    data = layer.to_dict()

    expected_matrix = Matrix.translation(5, 10) @ Matrix.scale(2, 3)

    assert data["type"] == "layer"
    assert data["name"] == "Test Layer"
    assert data["matrix"] == expected_matrix.to_list()
    assert data["visible"] is False
    assert data["stock_item_uid"] == "test-stock-uid"
    assert "children" in data
    assert len(data["children"]) == 2  # WorkPiece and Workflow


def test_layer_from_dict_deserialization():
    """Tests deserializing a Layer from a dictionary."""
    layer_dict = {
        "uid": "test-layer-uid",
        "type": "layer",
        "name": "Deserialized Layer",
        "matrix": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
        "visible": False,
        "stock_item_uid": "test-stock-uid",
        "children": [],
    }

    layer = Layer.from_dict(layer_dict)

    assert isinstance(layer, Layer)
    assert layer.uid == "test-layer-uid"
    assert layer.name == "Deserialized Layer"
    assert layer.matrix == Matrix.translation(10, 20)
    assert layer.visible is False
    assert layer.stock_item_uid == "test-stock-uid"


def test_layer_from_dict_with_no_children():
    """Tests deserializing a Layer with no children."""
    layer_dict = {
        "uid": "empty-layer-uid",
        "type": "layer",
        "name": "Empty Layer",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "visible": True,
        "stock_item_uid": None,
        "children": [],
    }

    layer = Layer.from_dict(layer_dict)

    assert isinstance(layer, Layer)
    assert layer.uid == "empty-layer-uid"
    assert layer.name == "Empty Layer"
    assert layer.visible is True
    assert layer.stock_item_uid is None
    assert len(layer.children) == 0
    assert layer.workflow is None


def test_layer_from_dict_ignores_unknown_child_types():
    """Tests that from_dict ignores children with unknown types."""
    layer_dict = {
        "uid": "mixed-layer-uid",
        "type": "layer",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "children": [
            {
                "uid": "workflow-uid",
                "type": "workflow",
            },
            {
                "uid": "unknown-uid",
                "type": "unknown_type",
            },
        ],
    }

    with patch(
        "rayforge.core.workflow.Workflow.from_dict"
    ) as mock_workflow_from_dict:
        mock_workflow = MagicMock()
        mock_workflow.get_local_bbox.return_value = None
        mock_workflow_from_dict.return_value = mock_workflow

        layer = Layer.from_dict(layer_dict)

        assert isinstance(layer, Layer)
        assert len(layer.children) == 1
        mock_workflow_from_dict.assert_called_once_with(
            {
                "uid": "workflow-uid",
                "type": "workflow",
            }
        )


def test_layer_roundtrip_serialization():
    """Tests that to_dict() and from_dict() produce equivalent objects."""
    # Create a layer with various properties
    original = Layer("Roundtrip Layer")
    original.matrix = Matrix.translation(5, 10) @ Matrix.scale(2, 3)
    original.visible = False
    original.stock_item_uid = "test-stock-uid"

    # Serialize and deserialize
    data = original.to_dict()
    restored = Layer.from_dict(data)

    # Check that the restored object has the same properties
    assert restored.uid == original.uid
    assert restored.name == original.name
    assert restored.matrix == original.matrix
    assert restored.visible == original.visible
    assert restored.stock_item_uid == original.stock_item_uid
    # Layer always has at least a workflow child
    assert len(restored.children) >= 1
