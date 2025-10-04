import pytest
from unittest.mock import MagicMock
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.import_source import ImportSource
from rayforge.core.vectorization_config import TraceConfig
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
    assert doc.import_sources == {}
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


def test_add_and_get_import_source(doc):
    """Tests the getter and setter for import sources."""
    source = ImportSource(
        source_file=Path("a.png"),
        original_data=b"abc",
        renderer=SvgRenderer(),
    )

    # Test adding a source
    doc.add_import_source(source)
    assert len(doc.import_sources) == 1
    assert source.uid in doc.import_sources

    # Test retrieving the source
    retrieved_source = doc.get_import_source_by_uid(source.uid)
    assert retrieved_source is source

    # Test retrieving a non-existent source
    assert doc.get_import_source_by_uid("non-existent-uid") is None

    # Test that adding a non-ImportSource object raises a TypeError
    with pytest.raises(TypeError):
        doc.add_import_source("not a source")


def test_add_layer_fires_descendant_added(doc):
    """Test adding a layer fires descendant_added with the layer as origin."""
    initial_layer_count = len(doc.layers)
    handler = MagicMock()
    doc.descendant_added.connect(handler)

    new_layer = Layer("Layer 2")
    doc.add_layer(new_layer)

    assert len(doc.layers) == initial_layer_count + 1
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
    handler.assert_called_once_with(doc, origin=step)


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
    handler.assert_called_once_with(doc, origin=step)


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
    handler.assert_called_once_with(doc, origin=step)


def test_doc_serialization_with_import_sources(doc):
    """Tests that the import_sources registry is serialized correctly."""
    # Source with vectorization config
    source1 = ImportSource(
        source_file=Path("a.png"),
        original_data=b"abc",
        renderer=SvgRenderer(),
        vector_config=TraceConfig(threshold=0.8),
    )
    # Source without vectorization config (e.g., an SVG)
    source2 = ImportSource(
        source_file=Path("b.svg"),
        original_data=b"def",
        renderer=SvgRenderer(),
    )
    doc.add_import_source(source1)
    doc.add_import_source(source2)

    data_dict = doc.to_dict()

    assert "import_sources" in data_dict
    assert len(data_dict["import_sources"]) == 2
    assert source1.uid in data_dict["import_sources"]
    assert source2.uid in data_dict["import_sources"]

    # Check structure of a source with config
    source1_dict = data_dict["import_sources"][source1.uid]
    assert source1_dict["uid"] == source1.uid
    assert source1_dict["source_file"] == "a.png"
    assert source1_dict["renderer_name"] == "SvgRenderer"
    assert source1_dict["vector_config"] is not None
    assert source1_dict["vector_config"]["threshold"] == 0.8

    # Check structure of a source without config
    source2_dict = data_dict["import_sources"][source2.uid]
    assert source2_dict["uid"] == source2.uid
    assert source2_dict["source_file"] == "b.svg"
    assert source2_dict["renderer_name"] == "SvgRenderer"
    assert source2_dict["vector_config"] is None


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
