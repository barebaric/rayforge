from typing import cast
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.source_asset import SourceAsset
from rayforge.image.svg.renderer import SvgRenderer
from rayforge.core.sketcher.sketch import Sketch


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
    assert doc.stock_assets == {}
    assert doc.stock_items == []
    assert doc.sketches == {}


@pytest.mark.parametrize(
    "asset_factory, compatibility_property_name",
    [
        (lambda: StockAsset(name="Test Stock"), "stock_assets"),
        (lambda: Sketch(name="Test Sketch"), "sketches"),
        (
            lambda: SourceAsset(
                source_file=Path("test.svg"),
                original_data=b"",
                renderer=SvgRenderer(),
            ),
            "source_assets",
        ),
    ],
)
def test_doc_asset_management(doc, asset_factory, compatibility_property_name):
    """Tests adding, removing, and getting all types of assets."""
    asset1 = asset_factory()
    asset2 = asset_factory()

    # 1. Test adding assets
    doc.add_asset(asset1)
    compatibility_property = getattr(doc, compatibility_property_name)
    assert len(doc.get_all_assets()) == 1
    assert len(compatibility_property) == 1
    assert asset1.uid in compatibility_property

    doc.add_asset(asset2)
    compatibility_property = getattr(doc, compatibility_property_name)
    assert len(doc.get_all_assets()) == 2
    assert len(compatibility_property) == 2

    # 2. Test getting asset by UID
    found_asset = doc.get_asset_by_uid(asset1.uid)
    assert found_asset is asset1
    assert doc.get_asset_by_uid("non-existent-uid") is None

    # 3. Test removing asset
    doc.remove_asset(asset1)
    compatibility_property = getattr(doc, compatibility_property_name)
    assert len(doc.get_all_assets()) == 1
    assert len(compatibility_property) == 1
    assert asset1.uid not in compatibility_property
    assert doc.get_asset_by_uid(asset1.uid) is None

    # 4. Test removing a non-existent asset (should not raise error)
    non_existent_asset = asset_factory()
    doc.remove_asset(non_existent_asset)
    assert len(doc.get_all_assets()) == 1

    # 5. Test that adding a non-IAsset object raises a TypeError
    with pytest.raises(TypeError):
        doc.add_asset("not an asset")


def test_doc_stock_items_management(doc):
    """Tests adding, removing, and getting stock items."""
    asset1 = StockAsset(name="Asset 1")
    doc.add_asset(asset1)
    stock1 = StockItem(stock_asset_uid=asset1.uid, name="Stock 1")
    stock2 = StockItem(stock_asset_uid=asset1.uid, name="Stock 2")

    doc.add_child(stock1)
    assert len(doc.stock_items) == 1
    assert stock1 in doc.stock_items
    assert stock1.parent is doc

    doc.add_child(stock2)
    assert len(doc.stock_items) == 2

    found_stock = doc.get_child_by_uid(stock1.uid)
    assert found_stock is stock1

    assert doc.get_child_by_uid("non-existent") is None

    doc.remove_child(stock1)
    assert len(doc.stock_items) == 1
    assert stock1 not in doc.stock_items
    assert stock1.parent is None


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
    """
    Tests that source assets are serialized correctly into the assets list.
    """
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
    doc.add_asset(asset1)
    doc.add_asset(asset2)

    data_dict = doc.to_dict()

    assert "assets" in data_dict
    all_assets = data_dict["assets"]
    source_asset_dicts = [a for a in all_assets if a.get("type") == "source"]
    assert len(source_asset_dicts) == 2

    uids_in_dict = {a["uid"] for a in source_asset_dicts}
    assert asset1.uid in uids_in_dict
    assert asset2.uid in uids_in_dict

    # Check structure of a source asset
    asset1_dict = next(a for a in source_asset_dicts if a["uid"] == asset1.uid)
    assert asset1_dict["uid"] == asset1.uid
    assert asset1_dict["source_file"] == "a.png"
    assert asset1_dict["renderer_name"] == "SvgRenderer"


def test_doc_serialization_with_stock(doc):
    """Tests that stock assets and items are serialized correctly."""
    asset1 = StockAsset(name="Asset 1")
    asset1.thickness = 10.0
    doc.add_asset(asset1)
    item1 = StockItem(stock_asset_uid=asset1.uid, name="Item 1")
    doc.add_child(item1)

    data_dict = doc.to_dict()

    assert "assets" in data_dict
    assert len(data_dict["assets"]) == 1
    asset1_dict = data_dict["assets"][0]
    assert asset1_dict["name"] == "Asset 1"
    assert asset1_dict["thickness"] == 10.0
    assert asset1_dict["type"] == "stock"

    assert "children" in data_dict
    # Find the stock item among the children (it's the second child after
    # the default layer)
    item1_dict = next(
        (c for c in data_dict["children"] if c.get("type") == "stockitem"),
        None,
    )
    assert item1_dict is not None
    assert item1_dict["name"] == "Item 1"
    assert item1_dict["stock_asset_uid"] == asset1.uid


def test_doc_serialization_with_sketches(doc):
    """Tests that the sketches registry is serialized correctly."""
    sketch1 = Sketch()
    sketch1.name = "My Sketch"
    doc.add_asset(sketch1)

    data_dict = doc.to_dict()

    assert "assets" in data_dict
    all_assets = data_dict["assets"]
    sketch_dicts = [a for a in all_assets if a.get("type") == "sketch"]

    assert len(sketch_dicts) == 1
    sketch1_dict = sketch_dicts[0]
    assert sketch1_dict["uid"] == sketch1.uid
    assert sketch1_dict["type"] == "sketch"
    assert sketch1_dict["name"] == "My Sketch"


def test_doc_from_dict_deserialization(doc):
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
        "assets": [],
    }

    with patch("rayforge.core.layer.Layer.from_dict") as mock_layer_from_dict:
        mock_layer = MagicMock()
        mock_layer.get_local_bbox.return_value = None
        mock_layer_from_dict.return_value = mock_layer

        new_doc = Doc.from_dict(doc_dict)

        assert isinstance(new_doc, Doc)
        assert len(new_doc.children) == 1
        assert new_doc.sketches == {}
        mock_layer_from_dict.assert_called_once_with(
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        )


def test_doc_from_dict_deserialization_modern_assets():
    """Tests deserializing a Doc with assets in the unified list."""
    doc_dict = {
        "uid": "test-doc-uid",
        "type": "doc",
        "children": [],
        "assets": [
            {
                "uid": "s1",
                "type": "stock",
                "name": "MDF",
                "geometry": {"commands": []},
            },
            {
                "uid": "k1",
                "type": "sketch",
                "name": "Circle",
                "params": {},
                "registry": {},
                "constraints": [],
                "origin_id": "origin-0",
            },
            {
                "uid": "r1",
                "type": "source",
                "name": "img.svg",
                "source_file": "img.svg",
                "original_data": b"",
                "renderer_name": "SvgRenderer",
            },
        ],
    }

    with patch.dict(
        "rayforge.image.renderer_by_name",
        {"SvgRenderer": SvgRenderer()},
        clear=True,
    ):
        new_doc = Doc.from_dict(doc_dict)

    assert len(new_doc.get_all_assets()) == 3
    assert len(new_doc.stock_assets) == 1
    assert len(new_doc.sketches) == 1
    assert len(new_doc.source_assets) == 1
    assert "s1" in new_doc.stock_assets
    assert "k1" in new_doc.sketches
    assert "r1" in new_doc.source_assets


def test_doc_deserialization_legacy_stock_format():
    """
    Tests that from_dict correctly handles the old format where stock items
    contained all their data.
    """
    legacy_doc_dict = {
        "uid": "legacy-doc",
        "stock_items": [
            {
                "uid": "stock-item-uid",
                "name": "Old Stock",
                "type": "stockitem",
                "thickness": 3.0,
                "material_uid": "mdf",
                "geometry": {"commands": []},
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
    }
    doc = Doc.from_dict(legacy_doc_dict)
    assert len(doc.stock_assets) == 1
    assert len(doc.stock_items) == 1

    item = doc.stock_items[0]
    asset = doc.get_asset_by_uid(item.stock_asset_uid)

    assert item.uid == "stock-item-uid"
    assert isinstance(asset, StockAsset)
    assert asset.name == "Old Stock"
    assert asset.thickness == 3.0
    assert asset.material_uid == "mdf"


def test_doc_deserialization_with_sketches():
    """Tests deserializing a doc that contains sketches using a mock."""
    sketch_data = {
        "uid": "sketch-123",
        "type": "sketch",
        "name": "Test",
        "params": {},
        "registry": {},
        "constraints": [],
        "origin_id": "origin-0",
    }
    doc_dict = {
        "uid": "test-doc-uid",
        "type": "doc",
        "active_layer_index": 0,
        "children": [],
        "stock_items": [],
        "source_assets": {},
        "sketches": {"sketch-123": sketch_data},
    }

    # Since we don't need to test Sketch.from_dict here, we can mock it
    with patch(
        "rayforge.core.sketcher.sketch.Sketch.from_dict"
    ) as mock_sketch_from_dict:
        mock_sketch = MagicMock(spec=Sketch)
        mock_sketch.uid = "sketch-123"
        # The mock needs asset_type_name for the .sketches property to work
        mock_sketch.asset_type_name = "sketch"
        mock_sketch_from_dict.return_value = mock_sketch

        doc = Doc.from_dict(doc_dict)

        mock_sketch_from_dict.assert_called_once_with(sketch_data)
        assert len(doc.sketches) == 1
        assert "sketch-123" in doc.sketches
        assert doc.get_asset_by_uid("sketch-123") is mock_sketch


def test_doc_roundtrip_serialization():
    """Tests that to_dict() and from_dict() produce equivalent objects."""
    # Create a document with layers
    original = Doc()
    original.uid = "test-doc-uid"

    # Add a second layer
    layer2 = Layer("Layer 2")
    original.add_layer(layer2)
    original.active_layer = layer2

    # Add a sketch
    sketch = Sketch()
    sketch.name = "My Test Sketch"
    original.add_asset(sketch)

    # Serialize and deserialize
    data = original.to_dict()
    restored = Doc.from_dict(data)

    # Check that the restored object has the same properties
    assert restored.uid == original.uid
    assert len(restored.layers) == len(original.layers)
    assert restored.uid == "test-doc-uid"
    assert len(restored.layers) == 2

    # Check that sketches were restored
    assert len(restored.sketches) == 1
    assert sketch.uid in restored.sketches
    restored_sketch = cast(Sketch, restored.get_asset_by_uid(sketch.uid))
    assert isinstance(restored_sketch, Sketch)
    assert restored_sketch.uid == sketch.uid
    assert restored_sketch.name == "My Test Sketch"


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
        "sketches": {},
    }

    with patch("rayforge.core.layer.Layer.from_dict") as mock_layer_from_dict:
        mock_layer = MagicMock()
        mock_layer.get_local_bbox.return_value = None
        mock_layer_from_dict.return_value = mock_layer

        doc = Doc.from_dict(doc_dict)

        assert doc._active_layer_index == 0  # Explicitly test the default
        mock_layer_from_dict.assert_called_once_with(
            {
                "uid": "layer1-uid",
                "type": "layer",
            }
        )


def test_has_result_logic(doc):
    """
    Tests the logic of the `has_result` method based on workpieces and
    the visibility of steps.
    """
    # 1. A new doc has no workpieces, so no result.
    assert not doc.has_result()

    # To test the step logic, we'll assume a workpiece exists.
    with patch.object(doc, "has_workpiece", return_value=True):
        # 2. With a workpiece but no steps, there's no result.
        assert not doc.has_result()

        # 3. Add an invisible step. Still no result.
        layer = doc.active_layer
        step = Step(typelabel="test")
        step.visible = False
        layer.workflow.add_step(step)
        assert not doc.has_result()

        # 4. Make the step visible. Now there is a result.
        step.visible = True
        assert doc.has_result()

        # 5. Add a second, invisible step. Still has a result because one is
        # visible.
        step2 = Step(typelabel="test2")
        step2.visible = False
        layer.workflow.add_step(step2)
        assert doc.has_result()

        # 6. Make the first step invisible. No result, as the only other step
        # is also invisible.
        step.visible = False
        assert not doc.has_result()

        # 7. Make the second step visible again. Result is back.
        step2.visible = True
        assert doc.has_result()


def test_remove_layer_adjusts_active_index(doc):
    """
    Tests that removing a layer correctly adjusts the active layer index.
    """
    layer1 = doc.layers[0]
    layer1.name = "Layer 1"
    layer2 = Layer("Layer 2")
    layer3 = Layer("Layer 3")
    doc.add_layer(layer2)
    doc.add_layer(layer3)
    doc.active_layer = layer2
    assert doc.active_layer is layer2
    assert doc._active_layer_index == 1

    # Case 1: Remove a layer before the active one
    doc.remove_layer(layer1)
    assert doc.layers == [layer2, layer3]
    assert doc._active_layer_index == 0  # Index shifts down
    assert doc.active_layer is layer2  # Instance remains the same

    # Reset for Case 2
    doc_case2 = Doc()
    l1 = doc_case2.layers[0]
    l1.name = "L1"
    l2 = Layer("L2")
    l3 = Layer("L3")
    doc_case2.add_layer(l2)
    doc_case2.add_layer(l3)
    doc_case2.active_layer = l2
    assert doc_case2._active_layer_index == 1

    # Case 2: Remove the active layer
    active_layer_changed_handler = MagicMock()
    doc_case2.active_layer_changed.connect(active_layer_changed_handler)
    doc_case2.remove_layer(l2)
    assert doc_case2.layers == [l1, l3]
    assert doc_case2._active_layer_index == 0  # Falls back to previous index
    assert doc_case2.active_layer is l1
    active_layer_changed_handler.assert_called_once()

    # Reset for Case 3
    doc_case3 = Doc()
    ly1 = doc_case3.layers[0]
    ly1.name = "LY1"
    ly2 = Layer("LY2")
    ly3 = Layer("LY3")
    doc_case3.add_layer(ly2)
    doc_case3.add_layer(ly3)
    doc_case3.active_layer = ly2
    assert doc_case3._active_layer_index == 1

    # Case 3: Remove a layer after the active one
    active_layer_changed_handler.reset_mock()
    doc_case3.active_layer_changed.connect(active_layer_changed_handler)
    doc_case3.remove_layer(ly3)
    assert doc_case3.layers == [ly1, ly2]
    assert doc_case3._active_layer_index == 1  # Index is unaffected
    assert doc_case3.active_layer is ly2
    active_layer_changed_handler.assert_not_called()


def test_remove_last_layer_is_noop(doc):
    """
    Tests that an attempt to remove the last layer from a document does
    nothing.
    """
    assert len(doc.layers) == 1
    last_layer = doc.layers[0]
    doc.remove_layer(last_layer)
    assert len(doc.layers) == 1
    assert doc.layers[0] is last_layer


def test_set_layers_preserves_stock_items(doc):
    """
    Tests that set_layers replaces layers but preserves other children like
    StockItems.
    """
    original_layer = doc.layers[0]
    asset = StockAsset(name="My Asset")
    doc.add_asset(asset)
    stock_item = StockItem(stock_asset_uid=asset.uid, name="My Stock")
    doc.add_child(stock_item)

    assert set(doc.children) == {original_layer, stock_item}

    new_layer1 = Layer("New Layer 1")
    new_layer2 = Layer("New Layer 2")

    # Act
    doc.set_layers([new_layer1, new_layer2])

    # Assert
    assert doc.layers == [new_layer1, new_layer2]  # Layers are replaced
    assert doc.stock_items == [stock_item]  # Stock item is preserved
    assert set(doc.children) == {new_layer1, new_layer2, stock_item}


def test_set_layers_raises_error_on_empty_list(doc):
    """
    Tests that set_layers raises a ValueError if called with an empty list.
    """
    with pytest.raises(ValueError):
        doc.set_layers([])


def test_set_layers_active_layer_logic(doc):
    """Tests how set_layers preserves or resets the active layer."""
    layer1 = doc.layers[0]
    layer1.name = "Layer 1"
    layer2 = Layer("Layer 2")
    layer3 = Layer("Layer 3")
    doc.add_layer(layer2)
    doc.add_layer(layer3)

    # Case 1: Active layer is preserved in the new list
    doc.active_layer = layer2
    doc.set_layers([layer3, layer1, layer2])
    assert doc.active_layer is layer2
    assert doc._active_layer_index == 2

    # Case 2: Active layer is removed from the new list
    doc.active_layer = layer2
    doc.set_layers([layer1, layer3])
    assert doc.active_layer is layer1  # Defaults to index 0
    assert doc._active_layer_index == 0


def test_update_stock_visibility(doc):
    """
    Tests that update_stock_visibility correctly shows/hides stock based
    on the active layer's assigned stock.
    """
    layer1 = doc.layers[0]
    layer2 = Layer("Layer 2")
    doc.add_layer(layer2)

    asset1 = StockAsset(name="Asset 1")
    asset2 = StockAsset(name="Asset 2")
    doc.add_asset(asset1)
    doc.add_asset(asset2)

    stock1 = StockItem(stock_asset_uid=asset1.uid, name="Wood")
    stock2 = StockItem(stock_asset_uid=asset2.uid, name="Acrylic")
    doc.add_child(stock1)
    doc.add_child(stock2)

    layer1.stock_item_uid = stock1.uid
    layer2.stock_item_uid = stock2.uid

    # Mock the method we want to check calls on
    stock1.set_visible = MagicMock()
    stock2.set_visible = MagicMock()

    # Act 1: set layer 2 as active (this is a CHANGE from default)
    doc.active_layer = layer2

    # Assert 1: stock1 should be hidden, stock2 visible
    stock1.set_visible.assert_called_once_with(False)
    stock2.set_visible.assert_called_once_with(True)

    stock1.set_visible.reset_mock()
    stock2.set_visible.reset_mock()

    # Act 2: set layer 1 as active (this is also a CHANGE)
    doc.active_layer = layer1

    # Assert 2: stock1 should be visible, stock2 hidden
    stock1.set_visible.assert_called_once_with(True)
    stock2.set_visible.assert_called_once_with(False)


def test_active_layer_setter_triggers_update_stock_visibility(doc):
    """
    Tests that changing the active layer automatically calls
    update_stock_visibility.
    """
    layer2 = Layer("Layer 2")
    doc.add_layer(layer2)

    with patch.object(doc, "update_stock_visibility") as mock_update:
        doc.active_layer = layer2
        mock_update.assert_called_once()

    with patch.object(doc, "update_stock_visibility") as mock_update:
        # Setting to the same layer should not trigger it again
        doc.active_layer = layer2
        mock_update.assert_not_called()


def test_from_dict_clears_default_layer():
    """
    Tests that Doc.from_dict removes the default layer created by __init__
    before populating from the dictionary.
    """
    doc_dict = {
        "uid": "test-doc-uid",
        "children": [
            {
                "uid": "layer1-uid",
                "type": "layer",
                "name": "Loaded Layer 1",
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "children": [],
                "workflow": {
                    "uid": "wf1",
                    "type": "workflow",
                    "name": "WF",
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "children": [],
                },
            },
            {
                "uid": "layer2-uid",
                "type": "layer",
                "name": "Loaded Layer 2",
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "children": [],
                "workflow": {
                    "uid": "wf2",
                    "type": "workflow",
                    "name": "WF",
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "children": [],
                },
            },
        ],
    }

    # Use real objects instead of mocks for a more robust test
    restored_doc = Doc.from_dict(doc_dict)

    assert len(restored_doc.layers) == 2
    layer_names = [layer.name for layer in restored_doc.layers]
    assert "Layer 1" not in layer_names  # Default name from __init__
    assert "Loaded Layer 1" in layer_names
    assert "Loaded Layer 2" in layer_names
