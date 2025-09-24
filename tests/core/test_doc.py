import pytest
from unittest.mock import MagicMock
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stocklayer import StockLayer
from rayforge.core.import_source import ImportSource
from rayforge.core.vectorization_config import TraceConfig
from rayforge.image.svg.renderer import SvgRenderer


@pytest.fixture
def doc():
    """Provides a real Doc instance. No mocks needed."""
    return Doc()


def test_doc_initialization(doc):
    """Verify a new Doc starts with a StockLayer and one regular Layer."""
    assert len(doc.children) == 2
    assert isinstance(doc.children[0], StockLayer)
    assert isinstance(doc.children[1], Layer)
    assert len(doc.layers) == 2  # .layers includes all Layer subclasses

    # Check that the first regular layer is active
    assert not isinstance(doc.active_layer, StockLayer)
    assert doc.active_layer.name == "Layer 1"
    assert doc.history_manager is not None
    assert doc.import_sources == {}


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
    step.set_power(500)

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


def test_doc_initial_state_has_stock_layer(doc):
    """A new document should always have a stock layer by default."""
    assert doc.stock_layer is not None
    assert isinstance(doc.stock_layer, StockLayer)


def test_doc_cannot_add_second_stocklayer(doc):
    """Test that adding a second StockLayer to a Doc raises a ValueError."""
    # A stock layer already exists from initialization
    with pytest.raises(ValueError):
        doc.add_child(StockLayer(name="Stock 2"))


def test_doc_cannot_remove_stocklayer(doc, caplog):
    """
    Test that attempting to remove the StockLayer does nothing and logs
    a warning.
    """
    stock_layer = doc.stock_layer
    assert stock_layer is not None
    assert stock_layer in doc.children

    # Act
    doc.remove_child(stock_layer)

    # Assert
    assert stock_layer in doc.children  # Still there
    assert "The StockLayer cannot be removed." in caplog.text


def test_doc_stock_layer_property(doc):
    """
    Test that the doc.stock_layer property correctly finds the stock layer.
    """
    stock_layer = doc.stock_layer
    assert stock_layer is not None
    assert stock_layer is doc.children[0]


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
