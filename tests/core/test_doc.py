import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.source_asset import SourceAsset
from rayforge.image.svg.renderer import SvgRenderer


@pytest.fixture
def doc():
    """Provides a real Doc instance. No mocks needed."""
    return Doc()


def test_doc_initialization(doc):
    """Verify a new Doc starts with one Layer."""
    assert len(doc.children) == 1
    assert isinstance(doc.children[0], Layer)
    assert len(doc.layers) == 1

    # Check that the first layer is active
    assert doc.active_layer.name == "Layer 1"
    assert doc.history_manager is not None
    assert doc.source_assets == {}
    assert doc.stock_items == []


def test_doc_stock_items_management(doc):
    """Tests adding, removing, and getting stock items."""
    stock1 = StockItem(name="Stock 1")
    stock2 = StockItem(name="Stock 2")

    # Test adding stock items
    doc.add_stock_item(stock1)
    assert len(doc.stock_items) == 1
    assert stock1 in doc.stock_items
    assert stock1.parent is doc

    doc.add_stock_item(stock2)
    assert len(doc.stock_items) == 2

    # Test getting stock item by UID
    found_stock = doc.get_stock_item_by_uid(stock1.uid)
    assert found_stock is stock1

    # Test getting non-existent stock item
    assert doc.get_stock_item_by_uid("non-existent") is None

    # Test removing stock item
    doc.remove_stock_item(stock1)
    assert len(doc.stock_items) == 1
    assert stock1 not in doc.stock_items
    assert stock1.parent is None


def test_add_and_get_source_asset(doc):
    """Tests the getter and setter for source assets."""
    asset = SourceAsset(
        source_file=Path("a.png"),
        original_data=b"abc",
        renderer=SvgRenderer(),
    )

    # Test adding a source
    doc.add_source_asset(asset)
    assert len(doc.source_assets) == 1
    assert asset.uid in doc.source_assets

    # Test retrieving the source
    retrieved_asset = doc.get_source_asset_by_uid(asset.uid)
    assert retrieved_asset is asset

    # Test retrieving a non-existent source
    assert doc.get_source_asset_by_uid("non-existent-uid") is None

    # Test that adding a non-SourceAsset object raises a TypeError
    with pytest.raises(TypeError):
        doc.add_source_asset("not a source")


def test_add_layer_fires_descendant_added(doc):
    """Test adding a layer fires descendant_added with the layer as origin."""
    initial_layer_count = len(doc.layers)
    handler = MagicMock()
    doc.descendant_added.connect(handler)

    new_layer = Layer("Layer 2")
    doc.add_layer(new_layer)

    assert len(doc.layers) == initial_layer_count + 1
    handler.assert_called_once_with(
        doc, origin=new_layer, parent_of_origin=doc
    )


def test_remove_layer_fires_descendant_removed(doc):
    """
    Test removing a layer fires descendant_removed with the layer as origin.
    """
    layer_to_remove = Layer("Layer 2")
    doc.add_layer(layer_to_remove)

    handler = MagicMock()
    doc.descendant_removed.connect(handler)
    doc.remove_layer(layer_to_remove)

    handler.assert_called_once_with(
        doc, origin=layer_to_remove, parent_of_origin=doc
    )


def test_descendant_updated_bubbles_up_to_doc(doc):
    """A descendant_updated signal from a Step should bubble up to the Doc."""
    handler = MagicMock()
    doc.descendant_updated.connect(handler)

    layer = (
        doc.active_layer
    )  # Use the active layer, which is guaranteed to be a regular one
    workflow = layer.workflow
    assert workflow is not None
    step = Step("Test Step")
    workflow.add_step(step)
    handler.reset_mock()  # Ignore the 'add' event

    # Act
    step.set_power(0.5)

    # Assert
    handler.assert_called_once_with(
        doc, origin=step, parent_of_origin=workflow
    )


def test_descendant_added_bubbles_up_to_doc(doc):
    """A descendant_added signal for a new Step should bubble up to the Doc."""
    handler = MagicMock()
    doc.descendant_added.connect(handler)

    layer = doc.active_layer
    workflow = layer.workflow
    assert workflow is not None
    step = Step("Test Step")

    # Act
    workflow.add_step(step)

    # Assert
    handler.assert_called_once_with(
        doc, origin=step, parent_of_origin=workflow
    )


def test_descendant_removed_bubbles_up_to_doc(doc):
    """A descendant_removed signal for a step should bubble up to the Doc."""
    layer = doc.active_layer
    workflow = layer.workflow
    assert workflow is not None
    step = Step("Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    doc.descendant_removed.connect(handler)

    # Act
    workflow.remove_step(step)

    # Assert
    handler.assert_called_once_with(
        doc, origin=step, parent_of_origin=workflow
    )


def test_doc_serialization_with_source_assets(doc):
    """Tests that the source_assets registry is serialized correctly."""
    asset1 = SourceAsset(
        source_file=Path("a.png"),
        original_data=b"abc",
        renderer=SvgRenderer(),
    )
    asset2 = SourceAsset(
        source_file=Path("b.svg"),
        original_data=b"def",
        renderer=SvgRenderer(),
    )
    doc.add_source_asset(asset1)
    doc.add_source_asset(asset2)

    data_dict = doc.to_dict()

    assert "source_assets" in data_dict
    assert len(data_dict["source_assets"]) == 2
    assert asset1.uid in data_dict["source_assets"]
    assert asset2.uid in data_dict["source_assets"]

    # Check structure of a source asset
    asset1_dict = data_dict["source_assets"][asset1.uid]
    assert asset1_dict["uid"] == asset1.uid
    assert asset1_dict["source_file"] == "a.png"
    assert asset1_dict["renderer_name"] == "SvgRenderer"


def test_doc_serialization_with_stock_items(doc):
    """Tests that stock_items are serialized correctly."""
    stock1 = StockItem(name="Stock 1")
    stock1.thickness = 10.0
    stock2 = StockItem(name="Stock 2")
    stock2.thickness = 15.0

    doc.add_stock_item(stock1)
    doc.add_stock_item(stock2)

    data_dict = doc.to_dict()

    assert "stock_items" in data_dict
    assert len(data_dict["stock_items"]) == 2

    # Check structure of stock items
    stock1_dict = data_dict["stock_items"][0]
    assert stock1_dict["name"] == "Stock 1"
    assert stock1_dict["thickness"] == 10.0

    stock2_dict = data_dict["stock_items"][1]
    assert stock2_dict["name"] == "Stock 2"
    assert stock2_dict["thickness"] == 15.0


def test_doc_from_dict_deserialization():
    """Tests deserializing a Doc from a dictionary."""
    doc_dict = {
        "uid": "test-doc-uid",
        "type": "doc",
        "active_layer_index": 0,
        "children": [
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        ],
        "stock_items": [],
        "source_assets": {},
    }

    with patch("rayforge.core.layer.Layer.from_dict") as mock_layer_from_dict:
        mock_layer = MagicMock()
        mock_layer.get_local_bbox.return_value = None
        mock_layer_from_dict.return_value = mock_layer

        doc = Doc.from_dict(doc_dict)

        assert isinstance(doc, Doc)
        assert len(doc.children) == 1
        mock_layer_from_dict.assert_called_once_with(
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        )


def test_doc_roundtrip_serialization():
    """Tests that to_dict() and from_dict() produce equivalent objects."""
    # Create a document with layers
    original = Doc()
    original.uid = "test-doc-uid"

    # Add a second layer
    layer2 = Layer("Layer 2")
    original.add_layer(layer2)
    original.active_layer = layer2

    # Serialize and deserialize
    data = original.to_dict()
    restored = Doc.from_dict(data)

    # Check that the restored object has the same properties
    assert restored.uid == original.uid
    assert len(restored.layers) == len(original.layers)
    assert restored.uid == "test-doc-uid"
    assert len(restored.layers) == 2


def test_doc_from_dict_with_default_active_layer_index():
    """Tests that from_dict uses default active_layer_index when missing."""
    doc_dict = {
        "uid": "test-doc-uid",
        "type": "doc",
        "children": [
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        ],
        "stock_items": [],
        "source_assets": {},
    }

    with patch("rayforge.core.layer.Layer.from_dict") as mock_layer_from_dict:
        mock_layer = MagicMock()
        mock_layer.get_local_bbox.return_value = None
        mock_layer_from_dict.return_value = mock_layer

        doc = Doc.from_dict(doc_dict)

        assert isinstance(doc, Doc)
        assert len(doc.children) == 1
        mock_layer_from_dict.assert_called_once_with(
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        )
