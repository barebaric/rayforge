import pytest

from rayforge.pipeline.producer.registry import ProducerRegistry
from rayforge.pipeline.producer import OpsProducer
from rayforge.pipeline.producer.placeholder import PlaceholderProducer
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem


class MockProducer(OpsProducer):
    """Mock producer for testing."""

    label = "Mock Producer"

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        generation_id,
        workpiece=None,
        settings=None,
        y_offset_mm=0.0,
        context=None,
    ):
        from rayforge.pipeline.artifact import WorkPieceArtifact

        return WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=(0.0, 0.0),
            generation_id=generation_id,
        )


class TestProducerRegistry:
    def test_register_and_get(self):
        registry = ProducerRegistry()
        registry.register(MockProducer)
        assert registry.get("MockProducer") == MockProducer

    def test_get_unknown_returns_none(self):
        registry = ProducerRegistry()
        assert registry.get("NonExistentProducer") is None

    def test_all_producers_returns_copy(self):
        registry = ProducerRegistry()
        registry.register(MockProducer)
        all_producers = registry.all_producers()
        assert all_producers == {"MockProducer": MockProducer}
        all_producers["NewOne"] = MockProducer
        assert "NewOne" not in registry.all_producers()

    def test_from_dict_unknown_type_returns_placeholder(self):
        data = {"type": "UnknownProducer", "params": {"foo": "bar"}}
        producer = OpsProducer.from_dict(data)
        assert isinstance(producer, PlaceholderProducer)
        assert producer.original_type == "UnknownProducer"
        assert producer.to_dict() == data

    def test_from_dict_missing_type_raises(self):
        data = {"params": {}}
        with pytest.raises(ValueError, match="must contain a 'type' key"):
            OpsProducer.from_dict(data)
