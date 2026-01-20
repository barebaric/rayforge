import unittest
from typing import Dict
from unittest.mock import MagicMock
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image.assembler import ItemAssembler
from rayforge.image.structures import LayoutItem
from rayforge.core.source_asset import SourceAsset
from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece


class TestItemAssembler(unittest.TestCase):
    def setUp(self):
        self.assembler = ItemAssembler()
        # Create a mock SourceAsset
        self.source = MagicMock(spec=SourceAsset)
        self.source.uid = "test-uid"
        self.spec = PassthroughSpec()
        self.mock_geo = MagicMock(spec=Geometry)

    def test_single_item_creation(self):
        """
        Verifies that a single layout item creates a WorkPiece.
        """
        plan = [
            LayoutItem(
                layer_id=None,
                layer_name=None,
                world_matrix=Matrix.identity(),
                normalization_matrix=Matrix.identity(),
                crop_window=(0, 0, 100, 100),
            )
        ]
        geometries: Dict[str | None, Geometry] = {None: self.mock_geo}

        items = self.assembler.create_items(
            self.source, plan, self.spec, "TestFile", geometries
        )

        self.assertEqual(len(items), 1)
        wp = items[0]
        # Assert proper type first
        self.assertIsInstance(wp, WorkPiece)
        self.assertEqual(wp.name, "TestFile")

        # Verify segment linkage (cast to WorkPiece to satisfy type checker)
        if isinstance(wp, WorkPiece) and wp.source_segment:
            self.assertEqual(wp.source_segment.source_asset_uid, "test-uid")
            self.assertEqual(
                wp.source_segment.crop_window_px, (0, 0, 100, 100)
            )
            self.assertIs(wp.source_segment.pristine_geometry, self.mock_geo)

    def test_split_layers_creation(self):
        """
        Verifies that layout items with layer_ids creates Layer objects
        containing WorkPieces.
        """
        mock_geo_a = MagicMock(spec=Geometry)
        mock_geo_b = MagicMock(spec=Geometry)
        plan = [
            LayoutItem(
                layer_id="LayerA",
                layer_name="LayerA",
                world_matrix=Matrix.identity(),
                normalization_matrix=Matrix.identity(),
                crop_window=(0, 0, 10, 10),
            ),
            LayoutItem(
                layer_id="LayerB",
                layer_name="LayerB",
                world_matrix=Matrix.translation(10, 10),
                normalization_matrix=Matrix.identity(),
                crop_window=(20, 20, 10, 10),
            ),
        ]
        geometries: Dict[str | None, Geometry] = {
            "LayerA": mock_geo_a,
            "LayerB": mock_geo_b,
        }

        items = self.assembler.create_items(
            self.source, plan, self.spec, "TestFile", geometries
        )

        self.assertEqual(len(items), 2)

        # Item 1 -> LayerA
        layer_a = items[0]
        self.assertIsInstance(layer_a, Layer)
        if isinstance(layer_a, Layer):
            self.assertEqual(layer_a.name, "LayerA")
            self.assertEqual(len(layer_a.workpieces), 1)
            wp_a = layer_a.workpieces[0]
            if wp_a.source_segment:
                self.assertEqual(wp_a.source_segment.layer_id, "LayerA")
                self.assertIs(
                    wp_a.source_segment.pristine_geometry, mock_geo_a
                )

        # Item 2 -> LayerB
        layer_b = items[1]
        self.assertIsInstance(layer_b, Layer)
        if isinstance(layer_b, Layer):
            self.assertEqual(layer_b.name, "LayerB")

            # Verify matrices applied to WorkPieces
            wp_b = layer_b.workpieces[0]
            if wp_b.source_segment:
                self.assertIs(
                    wp_b.source_segment.pristine_geometry, mock_geo_b
                )
            tx, ty = wp_b.matrix.get_translation()
            self.assertEqual(tx, 10.0)
            self.assertEqual(ty, 10.0)
