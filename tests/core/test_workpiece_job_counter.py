import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_counter import CounterSettings
from rayforge.core.workpiece import AnchorMode, WorkPiece
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
    def test_values_follow_doc_counter_and_offset(self, doc):
        doc.job_counter.counter = CounterSettings(current=10, step=5)
        wp = place(doc, "{counter}", offset=2)
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        assert provider.rendered[-1] == "20"

    def test_copy_added_with_new_offset_shows_its_value(self, doc):
        doc.job_counter.counter = CounterSettings(current=7)
        wp = place(doc, "{counter}")
        copy = wp.duplicate()
        assert isinstance(copy, WorkPiece)
        copy._variable_offset = 1
        doc.active_layer.add_child(copy)
        provider = wp.get_geometry_provider()
        assert isinstance(provider, TemplateProvider)
        assert provider.rendered[-1] == "8"

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

    def test_in_world_keeps_instance_params(self, doc):
        wp = place(doc, "{counter}")
        wp._geometry_provider_params = {"width": 5.0}
        assert wp.in_world().geometry_provider_params == {"width": 5.0}

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

    def test_counter_change_keeps_corner_in_place(self, doc):
        doc.job_counter.counter = CounterSettings(current=9)
        wp = place(doc, "{counter}")
        wp.pos = (10.0, 20.0)
        doc.job_counter.counter = CounterSettings(current=10)
        assert wp.pos == pytest.approx((10.0, 20.0))
        assert wp.size[0] == pytest.approx(4.0)

    def test_counter_change_keeps_center_in_place(self, doc):
        doc.job_counter.counter = CounterSettings(current=9)
        wp = place(doc, "{counter}")
        wp.anchor_mode = AnchorMode.CENTER
        wp.pos = (10.0, 20.0)
        center = wp.get_anchor_world()
        doc.job_counter.counter = CounterSettings(current=10)
        assert wp.get_anchor_world() == pytest.approx(center)
        assert wp.pos == pytest.approx((9.0, 20.0))

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
