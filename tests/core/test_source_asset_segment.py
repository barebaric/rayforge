from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.geo import Geometry
from rayforge.core.vectorization_spec import TraceSpec


class TestSourceAssetSegment:
    def test_clone_with_geometry(self):
        # Setup
        original_geo = Geometry()
        original_geo.move_to(0, 0)
        original_geo.line_to(1, 1)

        spec = TraceSpec(threshold=0.5)
        modifiers = [{"type": "blur", "radius": 2}]

        segment = SourceAssetSegment(
            source_asset_uid="uid-123",
            segment_mask_geometry=original_geo,
            vectorization_spec=spec,
            image_modifier_chain=modifiers,
        )

        # New geometry for clone
        new_geo = Geometry()
        new_geo.move_to(0, 0)
        new_geo.line_to(0.5, 0.5)

        # Execute
        clone = segment.clone_with_geometry(new_geo)

        # Verify basic properties
        assert clone.source_asset_uid == segment.source_asset_uid
        assert clone.segment_mask_geometry is new_geo
        assert clone.segment_mask_geometry is not original_geo

        # Verify Deep Copy of mutable fields
        assert clone.vectorization_spec == segment.vectorization_spec
        assert clone.vectorization_spec is not segment.vectorization_spec

        assert clone.image_modifier_chain == segment.image_modifier_chain
        assert clone.image_modifier_chain is not segment.image_modifier_chain

        # Modify clone, verify original matches
        # Use assert isinstance for type narrowing to satisfy static analysis
        assert isinstance(clone.vectorization_spec, TraceSpec)
        assert isinstance(segment.vectorization_spec, TraceSpec)

        clone.vectorization_spec.threshold = 0.9
        assert segment.vectorization_spec.threshold == 0.5

        clone.image_modifier_chain.append({"type": "invert"})
        assert len(segment.image_modifier_chain) == 1
