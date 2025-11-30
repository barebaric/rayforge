import pytest
from unittest.mock import MagicMock
from typing import List
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.split_cmd import SplitCmd, SplitStrategy
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.geo import Geometry
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec


class MockSplitStrategy(SplitStrategy):
    """
    A mock strategy that always returns 2 disjoint fragments for testing.
    """

    def calculate_fragments(self, workpiece: WorkPiece) -> List[Geometry]:
        g1 = Geometry()
        g1.move_to(0, 0)
        g1.line_to(0.2, 0.2)  # Increased size to pass >0.1mm dust filter

        g2 = Geometry()
        g2.move_to(0.5, 0.5)
        g2.line_to(0.7, 0.7)  # Increased size to pass >0.1mm dust filter

        return [g1, g2]


class SingleFragmentStrategy(SplitStrategy):
    """Strategy returning only 1 fragment (no split)."""

    def calculate_fragments(self, workpiece: WorkPiece) -> List[Geometry]:
        # Ensure we return a valid Geometry list, handling potential None
        geo = workpiece.boundaries
        if geo:
            return [geo]
        return []


@pytest.fixture
def mock_editor(context_initializer):
    task_manager = MagicMock()
    doc = Doc()
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def split_cmd(mock_editor):
    return SplitCmd(mock_editor)


@pytest.fixture
def workpiece_on_layer(mock_editor):
    """Creates a workpiece on the active layer."""
    layer = mock_editor.doc.active_layer

    # Create dummy geometry for the workpiece so it has something to split
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 1)

    segment = SourceAssetSegment(
        source_asset_uid="dummy",
        segment_mask_geometry=geo,
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece(name="TestItem", source_segment=segment)
    # Set dimensions directly on the workpiece
    wp.natural_width_mm = 10.0
    wp.natural_height_mm = 10.0
    wp.set_size(10.0, 10.0)

    layer.add_child(wp)
    return wp


def test_split_success(split_cmd, workpiece_on_layer, mock_editor):
    """
    Test that a workpiece is successfully split into multiple items,
    removed, and replaced in the document using the undo/redo system.
    """
    original_wp = workpiece_on_layer
    original_uid = original_wp.uid
    layer = original_wp.parent

    # Determine where the workpiece is in the layer.
    # The layer might contain a Workflow and other items.
    assert original_wp in layer.children
    initial_count = len(layer.children)

    new_items = split_cmd.split_items(
        [original_wp], strategy=MockSplitStrategy()
    )

    # Verify result
    assert len(new_items) == 2
    # The original is removed (-1) and two new ones added (+2), so net +1
    assert len(layer.children) == initial_count + 1
    assert original_wp not in layer.children
    for item in new_items:
        assert item in layer.children

    # Verify undo works
    mock_editor.history_manager.undo()
    assert len(layer.children) == initial_count

    # The workpiece should be back. Since we don't know the exact index (it
    # depends on add_child implementation/order), we search by UID.
    restored_wp = next(
        (c for c in layer.children if c.uid == original_uid), None
    )
    assert restored_wp is not None
    assert restored_wp.name == "TestItem"


def test_split_no_op(split_cmd, workpiece_on_layer):
    """Test that split does nothing if strategy returns <= 1 fragment."""
    original_wp = workpiece_on_layer
    layer = original_wp.parent
    initial_count = len(layer.children)

    new_items = split_cmd.split_items(
        [original_wp], strategy=SingleFragmentStrategy()
    )

    assert len(new_items) == 0
    assert len(layer.children) == initial_count
    assert original_wp in layer.children


def test_split_ignores_non_workpiece(split_cmd, mock_editor):
    """Test that split command ignores items that are not WorkPieces."""
    # Create a mock object that mimics a DocItem but isn't a WorkPiece
    non_wp_item = MagicMock()
    non_wp_item.parent = mock_editor.doc.active_layer
    # It needs to not be an instance of WorkPiece
    assert not isinstance(non_wp_item, WorkPiece)

    new_items = split_cmd.split_items([non_wp_item])

    assert len(new_items) == 0


def test_connectivity_strategy_real_split(split_cmd, mock_editor):
    """
    Test that the real ConnectivitySplitStrategy actually splits a disjoint
    geometry and correctly sizes/positions the fragments.
    """
    layer = mock_editor.doc.active_layer

    # Construct a geometry with two disjoint components in a normalized 0-1
    # space. Imagine a 30x10 bounding box.
    # Box 1: Left third (0-10mm -> 0.0-0.333 normalized)
    # Box 2: Right third (20-30mm -> 0.666-1.0 normalized)

    norm_geo = Geometry()
    # Box 1
    norm_geo.move_to(0.0, 0.0)
    norm_geo.line_to(0.333, 0.0)
    norm_geo.line_to(0.333, 1.0)
    norm_geo.line_to(0.0, 1.0)
    norm_geo.close_path()

    # Box 2
    norm_geo.move_to(0.666, 0.0)
    norm_geo.line_to(1.0, 0.0)
    norm_geo.line_to(1.0, 1.0)
    norm_geo.line_to(0.666, 1.0)
    norm_geo.close_path()

    segment = SourceAssetSegment(
        source_asset_uid="dummy",
        segment_mask_geometry=norm_geo,
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece(name="SplitMe", source_segment=segment)
    wp.natural_width_mm = 30.0
    wp.natural_height_mm = 10.0
    wp.set_size(30, 10)  # 30mm x 10mm
    wp.pos = (0, 0)
    layer.add_child(wp)

    # Execute Split
    # Uses default ConnectivitySplitStrategy
    new_items = split_cmd.split_items([wp])

    assert len(new_items) == 2

    # Sort by x position to be deterministic
    wps = sorted(new_items, key=lambda w: w.pos[0])

    # Check dimensions and positions
    # First piece: Should be ~10x10 at (0,0)
    assert wps[0].pos[0] == pytest.approx(0.0, abs=0.1)
    assert wps[0].size[0] == pytest.approx(10.0, abs=0.1)
    assert wps[0].size[1] == pytest.approx(10.0, abs=0.1)

    # Second piece: Should be ~10x10 at (20,0) (approx 19.98 due to 0.666)
    assert wps[1].pos[0] == pytest.approx(20.0, abs=0.1)
    assert wps[1].size[0] == pytest.approx(10.0, abs=0.1)
    assert wps[1].size[1] == pytest.approx(10.0, abs=0.1)
