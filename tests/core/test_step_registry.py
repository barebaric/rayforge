from rayforge.core.step_registry import StepRegistry, step_registry
from rayforge.core.step import Step
from rayforge.core.matrix import Matrix


class MockStep(Step):
    """Mock step subclass for testing."""

    def __init__(self, typelabel="Mock", name=None):
        super().__init__(typelabel=typelabel, name=name)
        self.mock_value = "default"

    @classmethod
    def create(cls, context=None, name=None, **kwargs):
        """Factory method for UI menu."""
        step = cls(name=name)
        step.mock_value = "created"
        return step


class TestStepRegistry:
    def test_register_and_get(self):
        registry = StepRegistry()
        registry.register(MockStep)
        assert registry.get("MockStep") == MockStep

    def test_get_unknown_returns_none(self):
        registry = StepRegistry()
        assert registry.get("NonExistentStep") is None

    def test_all_steps_returns_copy(self):
        registry = StepRegistry()
        registry.register(MockStep)
        all_steps = registry.all_steps()
        assert all_steps == {"MockStep": MockStep}
        all_steps["NewOne"] = MockStep
        assert "NewOne" not in registry.all_steps()

    def test_get_factories_returns_create_methods(self):
        registry = StepRegistry()
        registry.register(MockStep)
        factories = registry.get_factories()
        assert len(factories) == 1
        assert factories[0] == MockStep.create

    def test_get_factories_includes_all_non_hidden_steps(self):
        class AnotherStep(Step):
            def __init__(self):
                super().__init__(typelabel="Another")

            @classmethod
            def create(cls, context=None, name=None, **kwargs):
                return cls()

        registry = StepRegistry()
        registry.register(AnotherStep)
        factories = registry.get_factories()
        assert len(factories) == 1
        assert factories[0] == AnotherStep.create

    def test_get_factories_excludes_hidden_steps(self):
        class HiddenStep(Step):
            HIDDEN = True

            def __init__(self):
                super().__init__(typelabel="Hidden")

            @classmethod
            def create(cls, context=None, name=None, **kwargs):
                return cls()

        registry = StepRegistry()
        registry.register(HiddenStep)
        factories = registry.get_factories()
        assert len(factories) == 0

    def test_to_dict_includes_step_type(self):
        step = Step(typelabel="Test")
        data = step.to_dict()
        assert data["step_type"] == "Step"

    def test_to_dict_includes_subclass_name(self):
        step = MockStep(name="Test")
        data = step.to_dict()
        assert data["step_type"] == "MockStep"

    def test_from_dict_uses_registry_for_polymorphic_deserialization(self):
        step_registry.register(MockStep)
        try:
            step_dict = {
                "uid": "test-uid",
                "step_type": "MockStep",
                "name": "Test Mock",
                "typelabel": "Mock",
                "visible": True,
                "matrix": Matrix.identity().to_list(),
                "opsproducer_dict": None,
                "per_workpiece_transformers_dicts": [],
                "per_step_transformers_dicts": [],
            }

            step = Step.from_dict(step_dict)
            assert isinstance(step, MockStep)
            assert step.name == "Test Mock"
        finally:
            del step_registry._steps["MockStep"]

    def test_from_dict_fallback_to_base_step_for_unknown_type(self):
        step_dict = {
            "uid": "test-uid",
            "step_type": "UnknownStep",
            "name": "Test Unknown",
            "typelabel": "Unknown",
            "visible": True,
            "matrix": Matrix.identity().to_list(),
            "opsproducer_dict": None,
            "per_workpiece_transformers_dicts": [],
            "per_step_transformers_dicts": [],
        }

        step = Step.from_dict(step_dict)
        assert type(step) is Step
        assert step.name == "Test Unknown"

    def test_from_dict_backward_compatibility_without_step_type(self):
        step_dict = {
            "uid": "test-uid",
            "name": "Old Step",
            "typelabel": "OldType",
            "visible": True,
            "matrix": Matrix.identity().to_list(),
            "opsproducer_dict": None,
            "per_workpiece_transformers_dicts": [],
            "per_step_transformers_dicts": [],
        }

        step = Step.from_dict(step_dict)
        assert type(step) is Step
        assert step.name == "Old Step"
