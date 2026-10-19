import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_counter import CounterSettings
from rayforge.core.workpiece import WorkPiece
from tests.core.template_provider import TemplateProvider, place


@pytest.fixture
def doc():
    return Doc()


class TestVariableOffset:
    def test_default_offset_is_zero(self):
        assert WorkPiece("wp").variable_offset == 0

    def test_offset_round_trip(self):
        wp = WorkPiece("wp")
        wp.variable_offset = 4
        data = wp.to_dict()
        assert data["variable_offset"] == 4
        assert WorkPiece.from_dict(data).variable_offset == 4

    def test_zero_offset_is_not_serialized(self):
        assert "variable_offset" not in WorkPiece("wp").to_dict()

    def test_offset_is_not_an_extra_attribute(self):
        wp = WorkPiece("wp")
        wp.variable_offset = 2
        assert WorkPiece.from_dict(wp.to_dict()).extra == {}

    def test_duplicate_keeps_offset(self):
        wp = WorkPiece("wp")
        wp.variable_offset = 3
        copy = wp.duplicate()
        assert isinstance(copy, WorkPiece)
        assert copy.variable_offset == 3


class TestTemplateValues:
    def test_unparented_workpiece_has_no_values(self):
        assert WorkPiece("wp").template_values() == {}

    def test_values_follow_doc_counter_and_offset(self, doc):
        doc.job_counter.counter = CounterSettings(current=10, step=5)
        wp = place(doc, "{counter}", offset=2)
        assert wp.template_values() == {"counter": 20}

    def test_provider_receives_counter(self, doc):
        doc.job_counter.counter = CounterSettings(current=7)
        wp = place(doc, "No. {counter:03d}")
        wp.clear_render_cache()
        assert wp.boundaries is not None
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        assert provider.rendered[-1] == "No. 007"

    def test_instance_params_win_over_doc_values(self, doc):
        doc.job_counter.counter = CounterSettings(current=7)
        wp = place(doc, "{counter}")
        wp._geometry_provider_params = {"counter": 99}
        wp.clear_render_cache()
        assert wp.boundaries is not None
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        assert provider.rendered[-1] == "99"

    def test_in_world_bakes_values(self, doc):
        doc.job_counter.counter = CounterSettings(current=3)
        wp = place(doc, "{counter}", offset=1)
        world = wp.in_world()
        assert world.template_values() == {"counter": 4}

    def test_in_world_bakes_values_for_plain_templates(self, doc):
        doc.job_counter.counter = CounterSettings(current=3)
        wp = place(doc, "Hello")
        assert wp.in_world().template_values() == {"counter": 3}

    def test_in_world_keeps_instance_params(self, doc):
        wp = place(doc, "{counter}")
        wp._geometry_provider_params = {"width": 5.0}
        assert wp.in_world().geometry_provider_params == {"width": 5.0}

    def test_in_world_renders_baked_values(self, doc):
        doc.job_counter.counter = CounterSettings(current=3)
        wp = place(doc, "No. {counter}", offset=1)
        world = wp.in_world()
        world.clear_render_cache()
        provider = wp.get_geometry_provider()
        world._transient_geometry_provider = provider
        assert world.boundaries is not None
        assert isinstance(provider, TemplateProvider)
        assert provider.rendered[-1] == "No. 4"

    def test_in_world_survives_serialization(self, doc):
        doc.job_counter.counter = CounterSettings(current=8)
        wp = place(doc, "{counter}")
        world = WorkPiece.from_dict(wp.in_world().to_dict())
        assert world.template_values() == {"counter": 8}

    def test_provider_reports_used_variables(self, doc):
        wp = place(doc, "{counter:04d}")
        plain = place(doc, "Hello")
        counter_provider = wp.get_geometry_provider()
        plain_provider = plain.get_geometry_provider()
        assert isinstance(counter_provider, TemplateProvider)
        assert isinstance(plain_provider, TemplateProvider)
        assert counter_provider.variables_used() == {"counter"}
        assert plain_provider.variables_used() == set()


class TestRefreshOnChange:
    def test_counter_change_regenerates_text(self, doc):
        wp = place(doc, "{counter}")
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        doc.job_counter.counter = CounterSettings(current=1234)
        assert provider.rendered[-1] == "1234"

    def test_counter_change_keeps_scale(self, doc):
        doc.job_counter.counter = CounterSettings(current=9)
        wp = place(doc, "{counter}")
        wp.set_size(4.0, 10.0)
        doc.job_counter.counter = CounterSettings(current=10)
        width, height = wp.size
        # The text doubled in length; the 2x scale of the
        # workpiece is preserved.
        assert width == pytest.approx(8.0)
        assert height == pytest.approx(10.0)

    def test_counter_change_emits_updated(self, doc):
        wp = place(doc, "{counter}")
        received = []

        def on_updated(sender, **kwargs):
            received.append(sender)

        wp.updated.connect(on_updated)
        doc.job_counter.counter = CounterSettings(current=5)
        assert wp in received

    def test_unrelated_workpiece_keeps_its_text(self, doc):
        wp = place(doc, "Hello")
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        doc.job_counter.counter = CounterSettings(current=5)
        assert provider.rendered[-1] == "Hello"

    def test_offset_change_regenerates(self, doc):
        doc.job_counter.counter = CounterSettings(current=1)
        wp = place(doc, "{counter}")
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        wp.variable_offset = 9
        assert provider.rendered[-1] == "10"


class TestDocSerialization:
    def test_doc_round_trip_keeps_counter(self, doc):
        doc.job_counter.counter = CounterSettings(current=12, step=3)
        restored = Doc.from_dict(doc.to_dict())
        assert restored.job_counter.counter == doc.job_counter.counter

    def test_old_documents_get_default_counter(self, doc):
        data = doc.to_dict()
        data.pop("job_counter", None)
        restored = Doc.from_dict(data)
        assert restored.job_counter.counter == CounterSettings()

    def test_restored_doc_refreshes_on_change(self, doc):
        restored = Doc.from_dict(doc.to_dict())
        received = []

        def on_changed(sender):
            received.append(sender)

        restored.job_counter.changed.connect(on_changed)
        restored.job_counter.counter = CounterSettings(current=2)
        assert received == [restored.job_counter]

    def test_copies_count_uses_highest_offset(self, doc):
        place(doc, "{counter}", offset=0)
        place(doc, "{counter}", offset=5)
        place(doc, "Hello", offset=9)
        assert doc.job_counter_copies() == 6

    def test_copies_count_without_template_users(self, doc):
        place(doc, "Hello")
        assert doc.job_counter_copies() == 0
