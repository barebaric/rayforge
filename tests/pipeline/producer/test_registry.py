import pytest

from rayforge.pipeline.producer.registry import (
    ProducerRegistry,
    producer_registry,
)
from rayforge.pipeline.producer import (
    OpsProducer,
    ContourProducer,
    Rasterizer,
    DepthEngraver,
    DitherRasterizer,
    FrameProducer,
    MaterialTestGridProducer,
    ShrinkWrapProducer,
)
from rayforge.pipeline.producer.placeholder import PlaceholderProducer


class TestProducerRegistry:
    def test_register_and_get(self):
        registry = ProducerRegistry()
        registry.register(ContourProducer)
        assert registry.get("ContourProducer") == ContourProducer

    def test_get_unknown_returns_none(self):
        registry = ProducerRegistry()
        assert registry.get("NonExistentProducer") is None

    def test_all_producers_returns_copy(self):
        registry = ProducerRegistry()
        registry.register(ContourProducer)
        all_producers = registry.all_producers()
        assert all_producers == {"ContourProducer": ContourProducer}
        all_producers["NewOne"] = ContourProducer
        assert "NewOne" not in registry.all_producers()

    def test_global_registry_has_builtin_producers(self):
        assert producer_registry.get("ContourProducer") == ContourProducer
        assert producer_registry.get("Rasterizer") == Rasterizer
        assert producer_registry.get("DepthEngraver") == DepthEngraver
        assert producer_registry.get("DitherRasterizer") == DitherRasterizer
        assert producer_registry.get("FrameProducer") == FrameProducer
        assert (
            producer_registry.get("MaterialTestGridProducer")
            == MaterialTestGridProducer
        )
        assert (
            producer_registry.get("ShrinkWrapProducer") == ShrinkWrapProducer
        )

    def test_producer_by_name_backward_compatibility(self):
        from rayforge.pipeline.producer import producer_by_name

        assert producer_by_name.get("ContourProducer") == ContourProducer
        assert producer_by_name.get("Rasterizer") == Rasterizer

    def test_from_dict_uses_registry(self):
        data = {"type": "ContourProducer", "params": {}}
        producer = OpsProducer.from_dict(data)
        assert isinstance(producer, ContourProducer)

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
